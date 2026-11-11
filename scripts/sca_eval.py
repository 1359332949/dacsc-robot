"""Official-protocol rollout driver that can inject domain-invariant conditioning.

Executes the full action chunk every replan (same as the upstream LIBERO
evaluation) so baseline numbers reproduce the full official baseline.
mode=base : no module (used to verify protocol equality)
mode=sca  : StateConditionalAlignment with phi = [proprioceptive state, step_norm]
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
from dacsc.upstream import verify_release_checkpoint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--episodes-per-task", type=int, default=1)
    ap.add_argument("--mode", choices=["base", "sca", "srm", "off"], required=True)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
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
    module = None
    if args.mode == "off":
        if args.calibration is None:
            raise SystemExit("--calibration required for mode=off")
        from dacsc.offline_state_alignment import OfflineStateAlignment

        module = OfflineStateAlignment(
            policy.action_generator.blocks[0].attention,
            str(args.calibration),
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=horizon,
        )
    elif args.mode == "srm":
        if args.calibration is None:
            raise SystemExit("--calibration required for mode=srm")
        from dacsc.state_residual_mean import StateResidualMeanAlignment

        module = StateResidualMeanAlignment(
            policy.action_generator.blocks[0].attention,
            str(args.calibration),
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=horizon,
        )
    elif args.mode == "sca":
        if args.calibration is None:
            raise SystemExit("--calibration required for mode=sca")
        from dacsc.state_conditional_alignment import StateConditionalAlignment

        module = StateConditionalAlignment(
            policy.action_generator.blocks[0].attention,
            str(args.calibration),
            num_steps=int(policy.config["action_generator"]["num_sample_steps"]),
            action_horizon=horizon,
        )
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids: list[int] = []
    for part in args.tasks.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            task_ids.extend(range(int(lo), int(hi) + 1))
        else:
            task_ids.append(int(part))
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
                    np.random.seed(seed); torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    env.seed(seed); env.reset()
                    obs = env.set_init_state(initial_states[episode])
                    success, done, steps = False, False, 0
                    for _ in range(SETTLING_STEPS):
                        obs, reward, done, info = env.step(DUMMY_ACTION)
                        success = is_success(env, info, reward, done)
                        if success or done:
                            break
                    predict_index = 0
                    while not success and not done and steps < MAX_STEPS[args.suite]:
                        batch = policy_observation(obs, task.language, policy)
                        if module is not None:
                            state = batch["state"].detach().float().cpu().numpy().reshape(-1)
                            module.conditioning = np.concatenate([state, [min(predict_index / 15.0, 1.0)]])
                            module.reset()
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            with torch.no_grad():
                                chunk = policy.predict(batch)[0].float().cpu().numpy()
                        predict_index += 1
                        for action_value in chunk:
                            obs, reward, done, info = env.step(np.clip(action_value, -1.0, 1.0))
                            steps += 1
                            success = is_success(env, info, reward, done)
                            if success or done or steps >= MAX_STEPS[args.suite]:
                                break
                    record = {
                        "suite": args.suite,
                        "task_id": task_id,
                        "episode_id": episode,
                        "seed": seed,
                        "success": bool(success),
                        "policy_steps": steps,
                        "mode": args.mode,
                    }
                    output.write(json.dumps(record) + "\n"); output.flush()
                    print(f"{args.suite} task={task_id} ep={episode} success={int(success)} steps={steps}", flush=True)
            finally:
                env.close()
    if module is not None:
        print(f"[adaptive] mode_counts={module.mode_counts}", flush=True)
        module.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
