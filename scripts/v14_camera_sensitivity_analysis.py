"""Analyze controlled Camera-parameter sweep features."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def topk(a, k):
    return set(np.argsort(-np.asarray(a))[:k].tolist())


def subspace_angles(A, B, k):
    # orthonormal bases from PCA loadings
    def basis(X):
        X = X - X.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        return vt[:k].T
    U, V = basis(A), basis(B)
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return [float(np.degrees(np.arccos(np.clip(x, -1, 1)))) for x in s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()
    z = np.load(args.dir / "features.npz")
    m = json.loads((args.dir / "manifest.json").read_text())
    cfgs = [c["name"] for c in m["camera_configs"]]
    views = m["policy_views"]
    out = {"configs": [c for c in m["camera_configs"]], "k": args.k, "views": {}}
    for view in views:
        base = z[f"base_{view}"].astype(np.float64)
        deltas = {c: z[f"{c}_{view}"].astype(np.float64) - base for c in cfgs if c != "base"}
        pair = {c: np.abs(d).mean(axis=(0, 1)) for c, d in deltas.items()}
        signed = {c: d.mean(axis=(0, 1)) for c, d in deltas.items()}
        pairs = [pair[c] for c in pair]
        rho = np.eye(len(pair))
        names = list(pair)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rho[i, j] = rho[j, i] = spearmanr(pair[names[i]], pair[names[j]]).statistic
        overlap = {a: {b: len(topk(pair[a], args.k) & topk(pair[b], args.k)) / args.k
                       for b in names} for a in names}
        flat = {c: d.reshape(-1, d.shape[-1]) for c, d in deltas.items()}
        angles = {a: {b: subspace_angles(flat[a], flat[b], args.k)
                      for b in names if b != a} for a in names}
        # covariance eigen-spectrum of each variant and shared concatenated deltas
        spectra = {}
        for c in names:
            X = flat[c]
            cov = (X.T @ X) / max(len(X) - 1, 1)
            ev = np.linalg.eigvalsh(cov)[::-1]
            spectra[c] = (ev / max(ev.sum(), 1e-12))[:args.k].tolist()
        all_delta = np.concatenate([flat[c] for c in names], axis=0)
        all_delta = all_delta - all_delta.mean(axis=0, keepdims=True)
        _, sv, vt = np.linalg.svd(all_delta, full_matrices=False)
        energy = (sv ** 2) / max((sv ** 2).sum(), 1e-12)
        out["views"][view] = {
            "pair_mean": {c: float(pair[c].mean()) for c in names},
            "pair_top_dims": {c: np.argsort(-pair[c])[:args.k].tolist() for c in names},
            "signed_mean_norm": {c: float(np.linalg.norm(signed[c])) for c in names},
            "cross_config_spearman": dict(zip(names, rho.tolist())),
            "topk_overlap": overlap,
            "principal_angles_deg": angles,
            "eigen_energy_topk": spectra,
            "shared_delta_energy": energy[:args.k].tolist(),
            "shared_delta_dims": np.argsort(-np.abs(vt[0]))[:args.k].tolist(),
        }
    (args.dir / "sensitivity_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({v: {"pair_mean": out["views"][v]["pair_mean"],
                          "shared_energy": out["views"][v]["shared_delta_energy"]} for v in views}, indent=2))


if __name__ == "__main__":
    main()
