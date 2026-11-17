"""Offline per-view/token residual diagnostic for E2 captures.

Privileged diagnostic only: Clean/Shift pairing is used for analysis, never as
an estimator input.  No policy, GPU, or simulator is required.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-frac", type=float, default=0.6)
    args = ap.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    z = np.load(args.capture / "features.npz")
    rows = [r for r in manifest["records"] if r.get("compare_pass")]
    valid = [int(r["index"]) for r in rows]
    row = {int(r["index"]): int(r["feature_row"]) for r in rows}
    n = len(valid)
    split = int(round(args.source_frac * n))
    source_ids = valid[:split]
    eval_ids = valid[split:]
    out = {"capture": str(args.capture), "source_ids": source_ids, "eval_ids": eval_ids,
           "views": {}, "source_coverage": {}}
    state8 = z["state8"]
    for view in manifest["policy_views"]:
        clean = z[f"clean_{view}"]; shift = z[f"shift_{view}"]
        residuals = []
        for s in valid:
            i = row[s]
            residuals.append((shift[i] - clean[i]).mean(axis=0))
        residuals = np.stack(residuals)
        norms = np.linalg.norm(residuals, axis=1)
        # Temporal stability proxy: cosine between residuals from consecutive
        # capture states (these are ordered source-pairing states, not a rollout).
        cosines = [cosine(residuals[i], residuals[i + 1]) for i in range(len(valid) - 1)]
        # Per-token mean residual energy and temporal variance.
        token_energy = np.linalg.norm(residuals, axis=0)
        token_var = residuals.var(axis=0)
        stable_score = token_energy / (np.sqrt(token_var) + 1e-6)
        out["views"][view] = {
            "residual_norm_mean": float(norms.mean()),
            "residual_norm_std": float(norms.std()),
            "consecutive_cosine_mean": float(np.mean(cosines)),
            "consecutive_cosine_std": float(np.std(cosines)),
            "token_energy_top10_fraction": float(np.sort(token_energy)[-10:].sum() /
                                                 (token_energy.sum() + 1e-12)),
            "stable_token_score_top10_fraction": float(np.sort(stable_score)[-10:].sum() /
                                                       (stable_score.sum() + 1e-12)),
        }
    # Source coverage: nearest-source proprioception distance for eval states.
    x = state8[[row[s] for s in valid]]
    src = x[:split]; ev = x[split:]
    d_eval = np.linalg.norm(ev[:, None, :] - src[None, :, :], axis=-1)
    d_src = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=-1)
    np.fill_diagonal(d_src, np.inf)
    out["source_coverage"] = {
        "eval_to_source_min_mean": float(d_eval.min(axis=1).mean()),
        "eval_to_source_min_max": float(d_eval.min(axis=1).max()),
        "source_to_source_min_mean": float(d_src.min(axis=1).mean()),
        "source_to_source_min_max": float(d_src.min(axis=1).max()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
