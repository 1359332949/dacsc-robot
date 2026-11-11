"""Phase-1 SALT port driver on CoRP (no SG-SALT).

modes:
  frozen : no adaptation
  salt   : K-projection SALT with feature-change gate + flow-matching leftover loss
  self   : same update machinery, target = current detached prediction (paper control)

Protocol notes:
  * H=12 action horizon, E=`--exec-len` executed per replan, overlap = H-E.
  * only K rows of the fused qkv are adapted (SaltKeyAdapter); everything else frozen.
  * the previous NORMALISED chunk tail is cached; execution uses unnormalize_action.
  * beta and the optimizer persist inside an episode and are reset at its start.
  * the SALT gate is calibrated on separate nominal (clean) episodes and saved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import (  # noqa: E402
    DUMMY_ACTION, MAX_STEPS, SETTLING_STEPS, configure_libero, is_success, policy_observation,
)
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.salt_tta import FeatureChangeGate, SaltTTA, flow_matching_leftover_loss  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402


def build_env(args, bddl):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                              render_gpu_device_id=args.render_gpu)


def observed(args, policy, env, obs, task):
    batch = policy_observation(obs, task.language, policy)
    prepared = prepare_observation(batch, config=policy.config, normalizer=policy.normalizer,
                                   device=args.device)
    return prepared


def sample_normalised(policy, prepared, num_steps):
    with torch.no_grad():
        rep = policy.representation_extractor(**prepared)
        chunk = policy.action_generator.sample(rep, num_steps=num_steps)
    return rep, chunk


def calibrate(args, policy):
    """Collect gate statistics from nominal (clean) episodes, saved to --calibration."""
    from libero.libero import benchmark, get_libero_path
    gate = FeatureChangeGate()
    clean_root = Path(args.libero_src) / "libero" / "libero"
    changes = []
    for tid in [int(x) for x in args.calib_tasks.split(",")]:
        suite = benchmark.get_benchmark_dict()[args.suite]()
        task = suite.get_task(tid)
        bddl = clean_root / "bddl_files" / task.problem_folder / f"{task.name}.bddl"
        states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                            weights_only=False)
        env = build_env(args, bddl)
        seed = args.seed + tid * 1000
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        env.seed(seed); env.reset(); obs = env.set_init_state(states[0])
        for _ in range(SETTLING_STEPS):
            obs, _, done, _ = env.step(DUMMY_ACTION)
            if done: break
        gate.reset()
        steps = 0
        while steps < int(args.calib_steps):
            prepared = observed(args, policy, env, obs, task)
            with torch.no_grad():
                rep = policy.representation_extractor(**prepared)
            change = gate.token_change(rep.tokens)
            if gate.prev is not None and change.abs().sum() > 0:
                changes.append(change.detach().cpu().clone())
            _, chunk = sample_normalised(policy, prepared, args.num_steps)
            actions = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
            for a in actions[: args.exec_len]:
                obs, _, done, _ = env.step(np.clip(a, -1.0, 1.0))
                steps += 1
                if done or steps >= int(args.calib_steps): break
            if done: break
        env.close()
    gate.calibrate(changes)
    stats = {"median": gate.median.tolist(), "mad": gate.mad.tolist(), "z": gate.z,
             "mad_floor": gate.mad_floor, "clip": gate.clip, "num_tokens": gate.num_tokens}
    args.calibration.write_text(json.dumps(stats))
    print(f"[calib] episodes={len(args.calib_tasks.split(','))} frames={len(changes)} -> {args.calibration}",
          flush=True)


def load_gate(path: Path) -> FeatureChangeGate:
    blob = json.loads(path.read_text())
    gate = FeatureChangeGate(z=blob["z"], mad_floor=blob["mad_floor"], clip=blob["clip"],
                             num_tokens=blob["num_tokens"])
    gate.median = torch.tensor(blob["median"], dtype=torch.float32)
    gate.mad = torch.tensor(blob["mad"], dtype=torch.float32)
    return gate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--mode", choices=["frozen", "salt", "self"], default="frozen")
    ap.add_argument("--exec-len", type=int, default=6)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--adapt-steps", type=int, default=5)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--gate-mode", choices=["feature", "always"], default="feature")
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib-tasks", default="0,1")
    ap.add_argument("--calib-steps", default="120")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.calibrate and not args.tasks:
        ap.error("--tasks is required unless --calibrate")
    verify_release_checkpoint(_CORP_REPO / "checkpoints/corp_libero.pt")
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path

    policy = load_policy(_CORP_REPO / "checkpoints/corp_libero.pt", device=args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.calibrate:
        calibrate(args, policy)
        return 0
    gate = load_gate(args.calibration) if args.calibration is not None else None
    tta = None
    if args.mode in ("salt", "self"):
        tta = SaltTTA(policy, eta=args.eta, adapt_steps=args.adapt_steps, draws=args.draws)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [int(x) for x in args.tasks.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for task_id in task_ids:
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                weights_only=False)
            env = build_env(args, bddl)
            seed = args.seed + task_id * 1000
            np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            env.seed(seed); env.reset(); obs = env.set_init_state(states[0])
            for _ in range(SETTLING_STEPS):
                obs, _, done, _ = env.step(DUMMY_ACTION)
                if done: break
            if tta is not None:
                tta.reset()
            if gate is not None:
                gate.reset()
            success, done, steps = False, False, 0
            prev_norm = None
            updates = 0
            gate_step = -1
            predict_index = 0
            while not success and not done and steps < MAX_STEPS[args.suite]:
                prepared = observed(args, policy, env, obs, task)
                with torch.no_grad():
                    rep = policy.representation_extractor(**prepared)
                gate_active = False
                if args.gate_mode == "always":
                    gate_active = True
                    if gate_step < 0:
                        gate_step = predict_index
                elif gate is not None:
                    gate_active = gate.update(rep.tokens)
                    if gate_active and gate_step < 0:
                        gate_step = predict_index
                if tta is not None and gate_active and prev_norm is not None:
                    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                    for _ in range(args.adapt_steps):
                        target = prev_norm if args.mode == "salt" else None
                        if args.mode == "self":
                            with torch.no_grad():
                                target = policy.action_generator.sample(rep, num_steps=args.num_steps)[0].detach()
                        with torch.random.fork_rng(devices=devices):
                            torch.manual_seed(1234)
                            loss = flow_matching_leftover_loss(
                                policy.action_generator, rep, target, args.exec_len,
                                draws=args.draws, mode="leftover" if args.mode == "salt" else "self")
                        tta.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        tta.optimizer.step()
                        with torch.no_grad():
                            tta.adapter.beta.clamp_(-args.eta, args.eta)
                        updates += 1
                _, chunk = sample_normalised(policy, prepared, args.num_steps)
                prev_norm = chunk[0].detach()
                actions = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
                predict_index += 1
                for a in actions[: args.exec_len]:
                    obs, reward, done, info = env.step(np.clip(a, -1.0, 1.0))
                    steps += 1
                    success = is_success(env, info, reward, done)
                    if success or done or steps >= MAX_STEPS[args.suite]: break
            env.close()
            rec = {"suite": args.suite, "task_id": task_id, "success": bool(success), "steps": steps,
                   "mode": args.mode, "exec_len": args.exec_len, "updates": updates,
                   "gate_step": gate_step, "beta_norm": float(tta.adapter.beta.norm()) if tta else 0.0}
            output.write(json.dumps(rec) + "\n"); output.flush()
            print(rec, flush=True)
    if tta is not None:
        tta.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
