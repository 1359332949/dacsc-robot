"""Phase-1.1 SALT port driver (CoRP-native flow convention, SALT B.4 gate).

modes: frozen | salt | self
gate-mode: feature (SALT B.4, requires --calibration) | always (diagnostic ablation)
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

CLEAN_ROOT = Path("/hy-tmp/dacsc-robot-784fe2e/repos/CoRP/third_party/LIBERO/libero/libero")


def build_env(args, bddl):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                              render_gpu_device_id=args.render_gpu)


def observed(args, policy, env, obs, task):
    batch = policy_observation(obs, task.language, policy)
    return prepare_observation(batch, config=policy.config, normalizer=policy.normalizer,
                               device=args.device)


def sample_norm(policy, prepared, num_steps):
    with torch.no_grad():
        rep = policy.representation_extractor(**prepared)
        chunk = policy.action_generator.sample(rep, num_steps=num_steps)
    return rep, chunk


def run_episode(args, policy, env, task, seed, tta, gate, calibration_missing: bool):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    env.seed(seed); env.reset()
    from libero.libero import get_libero_path
    states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    states = torch.load(states_path, weights_only=False)
    obs = env.set_init_state(states[0])
    for _ in range(SETTLING_STEPS):
        obs, _, done, _ = env.step(DUMMY_ACTION)
        if done: break
    if tta is not None:
        tta.reset()
    if gate is not None:
        gate.reset()
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    draw_rng = torch.Generator(device=gen_device).manual_seed(seed + 17)
    success, done, steps = False, False, 0
    prev_norm = None
    updates = 0
    gate_step = -1
    predict_index = 0
    beta_trace = []
    loss_trace = []
    while not success and not done and steps < MAX_STEPS[args.suite]:
        prepared = observed(args, policy, env, obs, task)
        with torch.no_grad():
            rep = policy.representation_extractor(**prepared)
        gate_active = False
        if args.gate_mode == "always":
            gate_active = True
            if gate_step < 0: gate_step = predict_index
        elif gate is not None:
            gate_active = gate.update(rep.tokens)
            if gate_active and gate_step < 0: gate_step = predict_index
        if tta is not None and gate_active and prev_norm is not None:
            for _ in range(args.adapt_steps):
                if args.mode == "self":
                    with torch.no_grad():
                        target = policy.action_generator.sample(rep, num_steps=args.num_steps)[0].detach()
                    mode = "self"
                else:
                    target, mode = prev_norm, "leftover"
                loss = flow_matching_leftover_loss(policy.action_generator, rep, target,
                                                   args.exec_len, draws=args.draws, mode=mode,
                                                   rng=draw_rng)
                tta.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                tta.optimizer.step()
                with torch.no_grad():
                    tta.adapter.beta.clamp_(-args.eta, args.eta)
                updates += 1
                loss_trace.append(float(loss.detach()))
            beta_trace.append(tta.adapter.diagnostics())
        _, chunk = sample_norm(policy, prepared, args.num_steps)
        prev_norm = chunk[0].detach()
        actions = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
        predict_index += 1
        for a in actions[: args.exec_len]:
            obs, reward, done, info = env.step(np.clip(a, -1.0, 1.0))
            steps += 1
            success = is_success(env, info, reward, done)
            if success or done or steps >= MAX_STEPS[args.suite]: break
    summary = {}
    if beta_trace:
        summary = {
            "beta_l2_max": max(b["beta_l2"] for b in beta_trace),
            "beta_l2_final": beta_trace[-1]["beta_l2"],
            "beta_abs_mean_final": beta_trace[-1]["beta_abs_mean"],
            "beta_abs_max_final": beta_trace[-1]["beta_abs_max"],
            "frac_ge_095eta_final": beta_trace[-1]["frac_ge_095eta"],
            "frac_ge_099eta_final": beta_trace[-1]["frac_ge_099eta"],
            "loss_first": loss_trace[0] if loss_trace else None,
            "loss_last": loss_trace[-1] if loss_trace else None,
        }
    return {"success": bool(success), "steps": steps, "updates": updates,
            "gate_step": gate_step, "beta": summary}


def calibrate(args, policy):
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()[args.suite]()
    gate = FeatureChangeGate(num_tokens=args.gate_tokens)
    per_task: dict[int, list[list[torch.Tensor]]] = {}
    seeds, frames, episodes = [], 0, 0
    for tid in [int(x) for x in args.calib_tasks.split(",")]:
        task = suite.get_task(tid)
        bddl = CLEAN_ROOT / "bddl_files" / task.problem_folder / f"{task.name}.bddl"
        per_task[tid] = []
        for ep in range(args.calib_episodes):
            env = build_env(args, bddl)
            seed = args.seed + tid * 1000 + ep
            np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            env.seed(seed); env.reset()
            states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                weights_only=False)
            obs = env.set_init_state(states[ep % len(states)])
            for _ in range(SETTLING_STEPS):
                obs, _, done, _ = env.step(DUMMY_ACTION)
                if done: break
            gate.reset()
            changes: list[torch.Tensor] = []
            steps = 0
            while steps < int(args.calib_steps):
                prepared = observed(args, policy, env, obs, task)
                with torch.no_grad():
                    rep = policy.representation_extractor(**prepared)
                change = gate.token_change(rep.tokens)
                if gate.prev is not None:
                    changes.append(change.detach().cpu().clone())
                _, chunk = sample_norm(policy, prepared, args.num_steps)
                actions = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
                for a in actions[: args.exec_len]:
                    obs, _, done, _ = env.step(np.clip(a, -1.0, 1.0))
                    steps += 1
                    if done or steps >= int(args.calib_steps): break
                if done: break
            env.close()
            per_task[tid].append(changes)
            frames += len(changes)
            episodes += 1
            seeds.append(seed)
            print(f"[calib] task={tid} ep={ep} transitions={len(changes)}", flush=True)
    stats = gate.calibrate(per_task)
    payload = {"statistics": stats, "gate_tokens": args.gate_tokens,
               "calib_tasks": [int(x) for x in args.calib_tasks.split(",")],
               "calib_episodes": args.calib_episodes, "seeds": seeds,
               "frames": frames, "episodes": episodes,
               "checkpoint": str(_CORP_REPO / "checkpoints/corp_libero.pt"),
               "suite": args.suite, "exec_len": args.exec_len, "num_steps": args.num_steps}
    args.calibration.write_text(json.dumps(payload))
    print(f"[calib] threshold={stats['threshold']:.4f} frames={frames} episodes={episodes} -> {args.calibration}",
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--mode", choices=["frozen", "salt", "self"], default="frozen")
    ap.add_argument("--gate-mode", choices=["feature", "always"], default="feature")
    ap.add_argument("--gate-tokens", type=int, default=96, choices=[48, 96])
    ap.add_argument("--exec-len", type=int, default=6)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--adapt-steps", type=int, default=5)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib-tasks", default="0,1,2,3,4")
    ap.add_argument("--calib-episodes", type=int, default=4)
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
    if args.calibrate:
        calibrate(args, policy)
        return 0
    gate = None
    if args.gate_mode == "feature":
        if args.calibration is None or not args.calibration.exists():
            raise SystemExit("--calibration is required for gate-mode=feature")
        blob = json.loads(args.calibration.read_text())
        gate = FeatureChangeGate(num_tokens=args.gate_tokens)
        gate.load(blob["statistics"] | {"n_tokens": blob["gate_tokens"]})
    tta = SaltTTA(policy, eta=args.eta, adapt_steps=args.adapt_steps, draws=args.draws) \
        if args.mode in ("salt", "self") else None
    suite = benchmark.get_benchmark_dict()[args.suite]()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for task_id in [int(x) for x in args.tasks.split(",")]:
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            env = build_env(args, bddl)
            seed = args.seed + task_id * 1000
            rec = run_episode(args, policy, env, task, seed, tta, gate, False)
            env.close()
            rec.update({"suite": args.suite, "task_id": task_id, "mode": args.mode,
                        "gate_mode": args.gate_mode, "gate_tokens": args.gate_tokens,
                        "exec_len": args.exec_len, "seed": seed})
            output.write(json.dumps(rec) + "\n"); output.flush()
            print(rec, flush=True)
    if tta is not None:
        tta.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
