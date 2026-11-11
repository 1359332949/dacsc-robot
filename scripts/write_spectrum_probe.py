"""Measure how the cross-attention write spectrum changes under domain shift.

Decomposition (our method's view):
    u = V (w - mu_s)            support coordinates (V: source PCA basis)
    r = (w - mu_s) - V^T u      residual
    u_A = Q_A^T u               action-sensitive coordinates (Jacobian subspace)
    u_N = Q_N^T u               nuisance coordinates (complement inside support)

For each condition we report mean energies |u_A|^2, |u_N|^2, |r|^2 per token and
the per-coordinate variance ratios against the source support covariance.

The hook indexing is the verified one: call 0 is the prefix encoding over
representation tokens (never touched); calls 1..num_steps are the Euler steps
over action tokens; we capture at call == num_steps.
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
        if index == self.num_steps:
            self.value = output.detach().reshape(-1, output.shape[-1]).float().cpu()
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--chunks", type=int, default=2)
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
    mean_s = np.array(stats["mean"], dtype=np.float64)
    basis = np.array(stats["basis"], dtype=np.float64)            # [K, D]
    support_cov = np.array(stats["support_cov"], dtype=np.float64)
    action_dirs = np.array(stats["action_subspace"], dtype=np.float64)  # [D, r]
    action_in_support = basis @ action_dirs                        # [K, r]
    ua, sa, _ = np.linalg.svd(action_in_support, full_matrices=False)
    rank_a = int(np.sum(sa > max(sa) * 1e-6)) if sa.size else 0
    q_a = ua[:, :rank_a]
    full, _, _ = np.linalg.svd(q_a, full_matrices=True)
    q_n = full[:, q_a.shape[1]:]

    policy = load_policy(_CORP_REPO / "checkpoints/corp_libero.pt", device=args.device)
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
                for _ in range(args.chunks):
                    batch = policy_observation(observation, task.language, policy)
                    capture.reset()
                    with torch.no_grad():
                        chunk = policy.predict(batch)[0].float().cpu().numpy()
                    if capture.value is not None:
                        writes.append(capture.value.numpy().astype(np.float64))
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

    w = np.concatenate(writes, axis=0)                 # [N, D]
    centred = w - mean_s
    u = centred @ basis.T                              # [N, K]
    recon = u @ basis
    residual = centred - recon
    u_a = u @ q_a                                      # [N, r]
    u_n = u @ q_n                                      # [N, K-r]
    energy_a = float(np.mean(np.sum(u_a ** 2, axis=1)))
    energy_n = float(np.mean(np.sum(u_n ** 2, axis=1)))
    energy_r = float(np.mean(np.sum(residual ** 2, axis=1)))
    var_n = np.var(u_n, axis=0)
    lam_s_n = np.diag(q_n.T @ support_cov @ q_n)
    ratio_n = np.sqrt(np.clip(var_n / np.clip(lam_s_n, 1e-12, None), 1e-6, 1e6))
    tau_s = max(float(np.mean(np.sum(residual ** 2, axis=1))), 1e-12)
    residual_mean = np.mean(residual, axis=0)
    residual_cov = np.cov(residual, rowvar=False)
    residual_eigvals, residual_eigvecs = np.linalg.eigh(residual_cov)
    order = np.argsort(residual_eigvals)[::-1]
    residual_eigvals = residual_eigvals[order]
    residual_eigvecs = residual_eigvecs[:, order]
    payload = {
        "schema": "write_spectrum_v1",
        "residual_mean": residual_mean.tolist(),
        "residual_mean_norm": float(np.linalg.norm(residual_mean)),
        "residual_mean_energy": float(np.sum(residual_mean ** 2)),
        "residual_eigvals_top20": residual_eigvals[:20].tolist(),
        "residual_eigvals_sum": float(residual_eigvals.sum()),
        "residual_basis_topk": residual_eigvecs[:, :32].T.tolist(),   # [32, D]
        "residual_eigvals_topk": residual_eigvals[:32].tolist(),
        "tau_residual": float(energy_r),
        "n_frames": int(w.shape[0]),
        "energy_action_sensitive": energy_a,
        "energy_nuisance": energy_n,
        "energy_residual": energy_r,
        "nuisance_sigma_ratio_mean": float(np.mean(ratio_n)),
        "nuisance_sigma_ratio_median": float(np.median(ratio_n)),
        "residual_energy": energy_r,
        "nuisance_rank": int(q_n.shape[1]),
        "action_rank": int(q_a.shape[1]),
        "stats": str(args.stats),
        "tasks": parse_tasks(args.tasks),
    }
    args.out.write_text(json.dumps(payload))
    print(json.dumps({k: payload[k] for k in ("n_frames", "energy_action_sensitive", "energy_nuisance",
                                              "energy_residual", "nuisance_sigma_ratio_mean")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
