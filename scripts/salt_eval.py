"""Frozen vs SALT-style TTA evaluation on CoRP (leftover self-supervision).

mode=base : Frozen policy, execute exec_len of each chunk (same protocol)
mode=salt  : same rollout, but between replans run a few test-time Adam steps on
             the selected attention projections (bounded singular-value scaling)
             minimising MSE(chunk[:H-E], leftover); gate by disagreement.
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
from dacsc.salt_tta import SaltTTA  # noqa: E402
from dacsc.leftover_tta import sample_with_grad  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--mode", choices=["base", "salt"], required=True)
    ap.add_argument("--exec-len", type=int, default=6)
    ap.add_argument("--modules", default="projection", help="comma list: projection,qkv")
    ap.add_argument("--bound", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--adapt-steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    verify_release_checkpoint(_CORP_REPO / "checkpoints/corp_libero.pt")
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy = load_policy(_CORP_REPO / "checkpoints/corp_libero.pt", device=args.device)
    horizon = int(policy.config["action_horizon"])
    tta = None
    if args.mode == "salt":
        tta = SaltTTA(policy, [m.strip() for m in args.modules.split(",")], bound=args.bound, lr=args.lr)
        tta.optimizer = torch.optim.Adam(tta.params, lr=args.lr)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [int(x) for x in args.tasks.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for task_id in task_ids:
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
            seed = args.seed + task_id * 1000
            np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            env.seed(seed); env.reset(); obs = env.set_init_state(states[0])
            for _ in range(SETTLING_STEPS):
                obs, _, done, _ = env.step(DUMMY_ACTION)
                if done: break
            if tta is not None:
                tta.reset()
            success, done, steps = False, False, 0
            previous_chunk = None
            adapted_updates = 0
            while not success and not done and steps < MAX_STEPS[args.suite]:
                batch = policy_observation(obs, task.language, policy)
                if tta is not None and previous_chunk is not None:
                    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                    with torch.random.fork_rng(devices=devices):
                        for _ in range(args.adapt_steps):
                            torch.manual_seed(1234)
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                rep = policy.representation_extractor(
                                    batch["images"], batch["state"], batch["instruction"])
                                with torch.random.fork_rng(devices=devices):
                                    torch.manual_seed(1234)
                                    chunk = sample_with_grad(policy.action_generator, rep,
                                                             policy.config["action_generator"]["num_sample_steps"])[0].float()
                            target = torch.tensor(previous_chunk[args.exec_len:], device=chunk.device, dtype=chunk.dtype)
                            overlap = horizon - args.exec_len
                            loss = torch.nn.functional.mse_loss(chunk[:overlap], target[:overlap])
                            tta.optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            tta.optimizer.step()
                            adapted_updates += 1
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        chunk = policy.predict(batch)[0].float().cpu().numpy()
                previous_chunk = chunk
                for action in chunk[: args.exec_len]:
                    obs, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                    steps += 1
                    success = is_success(env, info, reward, done)
                    if success or done or steps >= MAX_STEPS[args.suite]: break
            env.close()
            rec = {"suite": args.suite, "task_id": task_id, "success": bool(success), "steps": steps,
                   "mode": args.mode, "exec_len": args.exec_len, "updates": adapted_updates}
            output.write(json.dumps(rec) + "\n"); output.flush()
            print(rec, flush=True)
    if tta is not None:
        tta.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
