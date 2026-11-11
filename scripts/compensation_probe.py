"""Test the compensation hypothesis.

For each rollout we record, at every replan's last Euler step, the write mean and
its projection onto the action-sensitive subspace A (and onto the residual):

    a_shift   = || P_A (mean_tokens(w) - mu_s) ||
    r_shift   = || (mean_tokens(w) - mu_s) - P_A(...) ||

If the shift in A is the policy's own compensation for the visual corruption,
successful episodes should show a LARGER a_shift than failed ones.
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
from evaluate_libero import DUMMY_ACTION, MAX_STEPS, SETTLING_STEPS, configure_libero, is_success, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402


class Capture:
    def __init__(self, attention, num_steps: int, horizon: int, mean_s: torch.Tensor, q_a: torch.Tensor):
        self.attention = attention
        self.original = attention.attend
        self.num_steps = num_steps
        self.horizon = horizon
        self.mean_s = mean_s
        self.q_a = q_a
        self.step = 0
        self.a_shifts: list[float] = []
        self.r_shifts: list[float] = []
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        if out.shape[1] != self.horizon:
            self.step = 0
            return out
        self.step += 1
        if self.step == self.num_steps:
            mean = out.detach().reshape(-1, out.shape[-1]).mean(dim=0).float()
            delta = (mean - self.mean_s.to(mean.device))
            proj = self.q_a.to(mean.device) @ (self.q_a.to(mean.device).T @ delta)
            self.a_shifts.append(float(proj.norm()))
            self.r_shifts.append(float((delta - proj).norm()))
        return out

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--episodes", type=int, default=2)
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

    stats = json.loads(args.stats.read_text())
    mean_s = torch.tensor(stats["mean"], dtype=torch.float32)
    action = torch.tensor(stats["action_subspace"], dtype=torch.float32)
    u, s, _ = torch.linalg.svd(action, full_matrices=False)
    rank = int(np.sum(s.cpu().numpy() > s[0].item() * 1e-8))
    q_a = u[:, :rank].contiguous()

    policy = load_policy(_CORP_REPO / "checkpoints/corp_libero.pt", device=args.device)
    capture = Capture(policy.action_generator.blocks[0].attention,
                      int(policy.config["action_generator"]["num_sample_steps"]),
                      int(policy.config["action_horizon"]), mean_s, q_a)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for task_id in [int(x) for x in args.tasks.split(",")]:
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            initial_states = torch.load(states_path, weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
            try:
                for episode in range(args.episodes):
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
                    capture.a_shifts.clear(); capture.r_shifts.clear()
                    while not success and not done and steps < MAX_STEPS[args.suite]:
                        batch = policy_observation(obs, task.language, policy)
                        capture.reset()
                        with torch.no_grad():
                            chunk = policy.predict(batch)[0].float().cpu().numpy()
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
                        "success": bool(success),
                        "n_predicts": len(capture.a_shifts),
                        "a_shift_mean": float(np.mean(capture.a_shifts)) if capture.a_shifts else 0.0,
                        "r_shift_mean": float(np.mean(capture.r_shifts)) if capture.r_shifts else 0.0,
                    }
                    output.write(json.dumps(record) + "\n"); output.flush()
                    print(record, flush=True)
            finally:
                env.close()
    capture.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
