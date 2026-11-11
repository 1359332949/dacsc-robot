"""Collect (domain-invariant conditioning, write) pairs from clean rollouts.

phi = [proprioceptive state (state_dim), normalized predict index] -- both are
domain-invariant: camera/background/noise perturbations change the image but not
the robot's proprioception or the timestep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402


class Capture:
    def __init__(self, attention, num_steps: int, horizon: int):
        self.attention = attention
        self.original = attention.attend
        self.num_steps = num_steps
        self.horizon = horizon
        self.step = 0
        self.write = None
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        if out.shape[1] != self.horizon:
            self.step = 0
            return out
        self.step += 1
        if self.step == self.num_steps:
            self.write = out.detach().reshape(-1, out.shape[-1]).mean(dim=0).float().cpu().numpy()
        return out

    def reset(self):
        self.step = 0
        self.write = None

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--max-predicts", type=int, default=25)
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
    capture = Capture(policy.action_generator.blocks[0].attention,
                      int(policy.config["action_generator"]["num_sample_steps"]),
                      int(policy.config["action_horizon"]))
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids: list[int] = []
    for part in args.tasks.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            task_ids.extend(range(int(lo), int(hi) + 1))
        else:
            task_ids.append(int(part))
    phi_list, write_list = [], []
    try:
        for task_id in task_ids:
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            initial_states = torch.load(states_path, weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
            try:
                seed = args.seed + task_id * 1000
                np.random.seed(seed); torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                env.seed(seed); env.reset()
                obs = env.set_init_state(initial_states[0])
                for _ in range(SETTLING_STEPS):
                    obs, reward, done, info = env.step(DUMMY_ACTION)
                for k in range(args.chunks):
                    batch = policy_observation(obs, task.language, policy)
                    state = batch["state"].detach().float().cpu().numpy().reshape(-1)
                    # MUST match the driver's normalization: min(predict_index/15, 1)
                    phi = np.concatenate([state, [min(k / 15.0, 1.0)]])
                    capture.reset()
                    with torch.no_grad():
                        chunk = policy.predict(batch)[0].float().cpu().numpy()
                    phi_list.append(phi.astype(np.float32)); write_list.append(capture.write.astype(np.float32))
                    for action_value in chunk:
                        obs, reward, done, info = env.step(np.clip(action_value, -1.0, 1.0))
                        if done:
                            break
                    if done:
                        break
            finally:
                env.close()
    finally:
        capture.close()
    phi = np.stack(phi_list); write = np.stack(write_list)
    np.savez(args.out, phi=phi, write=write)
    print(f"saved {phi.shape[0]} pairs, phi_dim={phi.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
