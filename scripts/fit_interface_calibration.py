"""Fit the ISC-AP closed-form calibration on unlabeled target writes.

Given source statistics (mean, basis V, support covariance, action subspace)
and a set of *unlabeled* calibration tasks from the target condition, estimate
the target write statistics and build the action-protected moment-matching
operator:

    u  = V (w - mu_s)                       # support coordinates
    M_N = Sigma_s,N^{1/2} Sigma_t,N^{-1/2}  # whitening-coloring on nuisance
    M   = Q_N M_N Q_N^T + Q_A Q_A^T         # identity on action subspace
    w' = w + V^T ( M (u - u_t) - u )

No labels are used and no model weights are changed.
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
    SETTLING_STEPS,
    configure_libero,
    policy_observation,
)
from corp import load_policy  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402


class Capture:
    def __init__(self, attention, num_steps: int):
        self.attention = attention
        self.original = attention.attend
        self.num_steps = num_steps
        self.call = 0
        self.value = None
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        index = self.call
        self.call += 1
        if (index + 1) % self.num_steps == 0:
            self.value = output.detach().reshape(-1, output.shape[-1]).mean(dim=0).float().cpu()
        return output

    def reset(self):
        self.call = 0
        self.value = None

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


def parse_tasks(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def sqrtm_psd(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, eps, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def invsqrtm_psd(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, eps, None)
    return (vectors / np.sqrt(values)) @ vectors.T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", required=True, help="unlabeled calibration task ids")
    ap.add_argument("--chunks-per-episode", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    ap.add_argument("--no-protection", action="store_true", help="ablation: calibrate the full support (no action mask)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    verify_release_checkpoint(Path("/hy-tmp/dacsc-robot/repos/CoRP/checkpoints/corp_libero.pt"))
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    stats = json.loads(args.stats.read_text())
    mean_s = np.array(stats["mean"], dtype=np.float64)
    basis = np.array(stats["basis"], dtype=np.float64)          # [K, D]
    support_cov_s = np.array(stats["support_cov"], dtype=np.float64)
    action_dirs = np.array(stats["action_subspace"], dtype=np.float64)  # [D, r]

    policy = load_policy(Path("/hy-tmp/dacsc-robot/repos/CoRP/checkpoints/corp_libero.pt"), device=args.device)
    num_steps = int(policy.config["action_generator"]["num_sample_steps"])
    capture = Capture(policy.action_generator.blocks[0].attention, num_steps)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    writes: list[np.ndarray] = []
    try:
        for task_id in parse_tasks(args.tasks):
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            initial_states = torch.load(states_path, weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
            try:
                seed = args.seed + task_id * 1000
                np.random.seed(seed)
                torch.manual_seed(seed)
                env.seed(seed)
                env.reset()
                observation = env.set_init_state(initial_states[0])
                for _ in range(SETTLING_STEPS):
                    observation, reward, done, info = env.step(DUMMY_ACTION)
                for _ in range(args.chunks_per_episode):
                    batch = policy_observation(observation, task.language, policy)
                    capture.reset()
                    with torch.no_grad():
                        chunk = policy.predict(batch)[0].float().cpu().numpy()
                    if capture.value is None:
                        break
                    writes.append(capture.value.numpy())
                    for action in chunk:
                        observation, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                        if done:
                            break
                    if done:
                        break
            finally:
                env.close()
    finally:
        capture.close()

    u_t = np.stack([(w - mean_s) @ basis.T for w in writes])          # [N, K]
    u_t_mean = u_t.mean(axis=0)
    cov_t = np.cov((u_t - u_t_mean), rowvar=False)
    action_in_support = basis @ action_dirs                            # [K, r]
    # orthonormal basis via SVD with an explicit rank threshold (QR alone does
    # not detect near-dependence when the action subspace is not fully inside
    # the support span)
    ua, sa, _ = np.linalg.svd(action_in_support, full_matrices=False)
    rank_a = int(np.sum(sa > max(sa) * 1e-6)) if sa.size else 0
    q_a = ua[:, :rank_a]
    # nuisance basis = orthonormal complement of the action subspace.
    # NOTE: taking vh[rank:] from the projector SVD returns the *range* of q_a
    # (i.e. the action subspace itself), not its complement -- that was a bug.
    full, _, _ = np.linalg.svd(q_a, full_matrices=True)
    q_n = full[:, q_a.shape[1]:]  # [K, K - r]
    cov_s_n = q_n.T @ support_cov_s @ q_n
    cov_t_n = q_n.T @ cov_t @ q_n
    # shrinkage + eigenvalue clipping: whitening-coloring with few target frames
    # is otherwise numerically explosive (Sigma_t,N nearly singular).
    def shrink(cov: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        k = cov.shape[0]
        return (1 - alpha) * cov + alpha * (np.trace(cov) / k) * np.eye(k)

    cov_s_n = shrink(cov_s_n)
    cov_t_n = shrink(cov_t_n)
    if args.no_protection:
        cov_s_all = shrink(support_cov_s)
        cov_t_all = shrink(cov_t)
        m = sqrtm_psd(cov_s_all) @ invsqrtm_psd(cov_t_all)
    else:
        m_n = sqrtm_psd(cov_s_n) @ invsqrtm_psd(cov_t_n)
        m = q_n @ m_n @ q_n.T + q_a @ q_a.T
    # clip the calibration eigenvalues to a safe range (soft correction)
    values, vectors = np.linalg.eigh(m)
    values = np.clip(values, 0.5, 2.0)
    m = (vectors * values) @ vectors.T

    payload = {
        "schema": "interface_calibration_v1",
        "source_stats": str(args.stats),
        "suite": args.suite,
        "calibration_tasks": parse_tasks(args.tasks),
        "n_frames": len(writes),
        "mean": mean_s.tolist(),
        "basis": basis.tolist(),
        "transform": m.tolist(),
        "target_mean": u_t_mean.tolist(),
        "action_rank_in_support": int(q_a.shape[1]),
        "nuisance_rank": int(q_n.shape[1]),
    }
    args.out.write_text(json.dumps(payload))
    print(json.dumps({"frames": len(writes), "action_rank": int(q_a.shape[1]), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
