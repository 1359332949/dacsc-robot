"""ISA rollout driver (modified protocol: execute exec_len < horizon to keep a leftover).

Baseline mode uses exactly the same loop without adaptation, so the comparison
is paired under the same protocol. Official LIBERO evaluation executes the full
chunk; this deviation must be disclosed.
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
    DUMMY_ACTION,
    MAX_STEPS,
    SETTLING_STEPS,
    configure_libero,
    is_success,
    policy_observation,
)
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.interface_anchoring import InterfaceAnchoring  # noqa: E402
from dacsc.leftover_tta import leftover_loss, sample_with_grad  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=_CORP_REPO / "checkpoints/corp_libero.pt")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--episodes-per-task", type=int, default=1)
    ap.add_argument("--exec-len", type=int, default=4)
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--beta-max", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--adapt-steps", type=int, default=5)
    ap.add_argument("--lambda-src", type=float, default=0.1)
    ap.add_argument("--lambda-reg", type=float, default=1e-3)
    ap.add_argument("--mode", choices=["base", "tta"], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    ap.add_argument("--log-adapt", action="store_true")
    ap.add_argument("--residual-gate", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    verify_release_checkpoint(args.checkpoint)
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy = load_policy(args.checkpoint, device=args.device)
    stats = None
    adapter = None
    if args.mode == "tta":
        if args.stats is None:
            raise SystemExit("--stats is required for --mode tta")
        stats = json.loads(args.stats.read_text())
        adapter = InterfaceAnchoring(
            policy.action_generator.blocks[0].attention,
            stats,
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            beta_max=args.beta_max,
            lr=args.lr,
            steps=args.adapt_steps,
            lambda_src=args.lambda_src,
            lambda_reg=args.lambda_reg,
        )

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [int(item) for item in args.tasks.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for task_id in task_ids:
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            initial_states = torch.load(states_path, weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
            try:
                for episode in range(args.episodes_per_task):
                    seed = args.seed + task_id * 1000 + episode
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    env.seed(seed)
                    env.reset()
                    observation = env.set_init_state(initial_states[episode])
                    success, done, steps = False, False, 0
                    for _ in range(SETTLING_STEPS):
                        observation, reward, done, info = env.step(DUMMY_ACTION)
                        success = is_success(env, info, reward, done)
                        if success or done:
                            break
                    previous_chunk = None
                    while not success and not done and steps < MAX_STEPS[args.suite]:
                        batch = policy_observation(observation, task.language, policy)
                        if adapter is not None and previous_chunk is not None:
                            prepared = prepare_observation(
                                batch, config=policy.config, normalizer=policy.normalizer, device=args.device
                            )
                            target = torch.tensor(
                                previous_chunk[args.exec_len:], device=args.device, dtype=torch.float32
                            )
                            trace = []
                            # isolate adaptation RNG so the execution chunk is drawn
                            # from the SAME noise as the un-adapted baseline
                            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                            rng_context = torch.random.fork_rng(devices=devices)
                            rng_context.__enter__()
                            for step_index in range(args.adapt_steps):
                                adapter.active = True
                                adapter.reset()
                                torch.manual_seed(seed)
                                representation = policy.representation_extractor(
                                    prepared["images"], prepared["state"], prepared["instruction"]
                                )
                                chunk = sample_with_grad(
                                    policy.action_generator, representation,
                                    int(policy.config["action_generator"]["num_sample_steps"]),
                                )[0].float()
                                loss_action = leftover_loss(chunk, target, args.exec_len)
                                if step_index == 0 and args.residual_gate is not None:
                                    overlap = chunk.shape[0] - args.exec_len
                                    residual = float(torch.mean(torch.abs(chunk[:overlap] - target[:overlap])))
                                    if residual < args.residual_gate:
                                        if args.log_adapt:
                                            print(
                                                f"  gate-skip task={task_id} residual={residual:.4f} "
                                                f"< {args.residual_gate}",
                                                flush=True,
                                            )
                                        break
                                loss = loss_action + args.lambda_reg * adapter.beta.square().sum()
                                if adapter.source_loss is not None:
                                    loss = loss + args.lambda_src * adapter.source_loss
                                adapter.optimizer.zero_grad(set_to_none=True)
                                loss.backward()
                                adapter.optimizer.step()
                                with torch.no_grad():
                                    adapter.beta.clamp_(-args.beta_max, args.beta_max)
                                trace.append(float(loss.detach()))
                            rng_context.__exit__(None, None, None)
                            if args.log_adapt:
                                overlap = chunk.shape[0] - args.exec_len
                                delta = float(torch.mean(torch.abs(chunk[:overlap] - target[:overlap])))
                                print(
                                    f"  adapt replan task={task_id} beta_max={float(adapter.beta.abs().max()):.3f} "
                                    f"loss_action={trace[-1]:.4f} anchor={float(adapter.source_loss.detach()):.3f} "
                                    f"leftover_residual={delta:.4f}",
                                    flush=True,
                                )
                        if adapter is not None:
                            adapter.active = True
                            adapter.reset()
                            with torch.no_grad():
                                chunk = policy.predict(batch)[0].float().cpu().numpy()
                            adapter.active = False
                        else:
                            with torch.no_grad():
                                chunk = policy.predict(batch)[0].float().cpu().numpy()
                        previous_chunk = chunk
                        for action in chunk[: args.exec_len]:
                            observation, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                            steps += 1
                            success = is_success(env, info, reward, done)
                            if success or done or steps >= MAX_STEPS[args.suite]:
                                break
                    record = {
                        "suite": args.suite,
                        "task_id": task_id,
                        "episode_id": episode,
                        "seed": seed,
                        "success": success,
                        "policy_steps": steps,
                        "mode": args.mode,
                        "exec_len": args.exec_len,
                    }
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    print(f"{args.suite} task={task_id} ep={episode} success={int(success)} steps={steps}", flush=True)
            finally:
                env.close()
    if adapter is not None:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
