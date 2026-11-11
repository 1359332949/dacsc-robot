"""Measure action sensitivity of each source residual direction.

For every frame we push the last-Euler-step write along one residual direction
d_i (unit norm in write space) and record ||Delta action|| / eps.  High values
mark residual directions that the action head actually uses; the suppression
weight can then be (1 - normalised sensitivity) so that only low-sensitivity
(noise-like) directions are shrunk.
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
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402


class PushHook:
    def __init__(self, attention, num_steps: int, horizon: int):
        self.attention = attention
        self.original = attention.attend
        self.num_steps = num_steps
        self.horizon = horizon
        self.step = 0
        self.direction = None
        self.eps = 0.0
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        if out.shape[1] != self.horizon:
            self.step = 0
            return out
        self.step += 1
        if self.step == self.num_steps and self.direction is not None:
            out = out + self.eps * self.direction
        return out

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--spectrum", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--frames", type=int, default=2)
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

    spectrum = json.loads(args.spectrum.read_text())
    basis = np.array(spectrum["residual_basis_topk"][: args.k], dtype=np.float64)  # [k, D]
    policy = load_policy(_CORP_REPO / "checkpoints/corp_libero.pt", device=args.device)
    num_steps = int(policy.config["action_generator"]["num_sample_steps"])
    horizon = int(policy.config["action_horizon"])
    hook = PushHook(policy.action_generator.blocks[0].attention, num_steps, horizon)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    responses = []
    try:
        for task_id in [int(x) for x in args.tasks.split(",")]:
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            initial_states = torch.load(states_path, weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
            try:
                seed = args.seed + task_id * 1000
                np.random.seed(seed); torch.manual_seed(seed); env.seed(seed)
                env.reset(); observation = env.set_init_state(initial_states[0])
                for _ in range(SETTLING_STEPS):
                    observation, reward, done, info = env.step(DUMMY_ACTION)
                for _ in range(args.frames):
                    batch = policy_observation(observation, task.language, policy)
                    hook.reset()
                    with torch.no_grad():
                        base = policy.predict(batch)[0].float().cpu().numpy()
                    frame = []
                    scale = float(np.linalg.norm(basis[0]))
                    for i in range(args.k):
                        direction = torch.tensor(basis[i] / (scale + 1e-8), dtype=torch.float32, device=args.device)
                        hook.reset()
                        hook.direction = direction
                        hook.eps = 0.05
                        with torch.no_grad():
                            pushed = policy.predict(batch)[0].float().cpu().numpy()
                        hook.direction = None
                        frame.append(float(np.linalg.norm(pushed - base)) / hook.eps)
                    responses.append(frame)
                    with torch.no_grad():
                        chunk = policy.predict(batch)[0].float().cpu().numpy()
                    for action in chunk:
                        observation, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                        if done:
                            break
                    if done:
                        break
            finally:
                env.close()
    finally:
        hook.close()

    arr = np.array(responses)  # [frames, k]
    payload = {
        "schema": "residual_sensitivity_v1",
        "k": args.k,
        "n_frames": int(arr.shape[0]),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "normalised": (arr.mean(axis=0) / max(arr.mean(axis=0).max(), 1e-9)).tolist(),
    }
    args.out.write_text(json.dumps(payload))
    print(json.dumps({"frames": int(arr.shape[0]), "mean_top5": [round(x, 3) for x in payload["mean"][:5]],
                      "mean_last5": [round(x, 3) for x in payload["mean"][-5:]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
