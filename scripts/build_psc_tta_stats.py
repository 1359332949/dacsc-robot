"""Build PSC (full) source statistics for the released CoRP LIBERO policy.

PSC (arXiv:2603.09338) keeps source statistics {mu_s, V_s, Lambda_s, tau} fixed
at test time and updates only affine parameters of normalization layers.
This script fits those statistics on clean LIBERO features captured by
scripts/representation_probe_capture.py (feature = mean over representation
tokens).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weight-bias", type=float, default=1.0)
    ap.add_argument("--weight-exp", type=float, default=1.0)
    args = ap.parse_args()
    data = np.load(args.capture, allow_pickle=True)
    features = data["reps"].astype(np.float64)
    mean = features.mean(axis=0)
    centred = features - mean
    covariance = np.cov(centred, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    rank = min(args.rank, features.shape[1] - 1)
    basis = eigenvectors[:, :rank].T  # [K, D]
    lambdas = np.clip(eigenvalues[:rank], 1e-8, None)
    tau = float(np.clip(eigenvalues[rank:].mean(), 1e-8, None))
    # Prefer the released head weights for the probe weights when available; the
    # released CoRP head is a nonlinear flow expert, so use the linear action
    # sensitivity (ridge probe) as the paper's head-vector proxy.
    actions = data["actions"].astype(np.float64)
    x = features - mean
    y = actions - actions.mean(axis=0, keepdims=True)
    lam = 1e-2
    w = np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)  # [D, A]
    a = basis @ w  # [K, A]
    probes = []
    eye = np.eye(rank)
    probes.extend(eye)
    for i in range(rank):
        for j in range(i + 1, rank):
            probes.append((eye[i] + eye[j]) / np.sqrt(2))
            probes.append((eye[i] - eye[j]) / np.sqrt(2))
    probes = np.stack(probes)  # [K^2, K]
    q_weights = (np.abs(probes @ a).mean(axis=1) + args.weight_bias) ** args.weight_exp
    payload = {
        "schema": "psc_tta_stats_v1",
        "model": "official CoRP LIBERO checkpoint",
        "feature": "mean over representation tokens",
        "rank": rank,
        "n_frames": int(features.shape[0]),
        "mean": mean.tolist(),
        "basis": basis.tolist(),
        "lambdas": lambdas.tolist(),
        "tau": tau,
        "participation_ratio": float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()),
        "probes": probes.tolist(),
        "probe_weights": q_weights.tolist(),
        "kl_eps": 1e-8,
        "reference": "SSA official code (kzkadc/regression-tta) loss/optimizer conventions",
    }
    args.out.write_text(json.dumps(payload))
    print(json.dumps({"rank": rank, "tau": tau, "n": int(features.shape[0])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
