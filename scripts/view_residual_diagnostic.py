"""Offline per-view and patch-token residual diagnostic for E2 captures.

Privileged diagnostic only: Clean/Shift pairing is used for analysis, never as
an estimator input.  No policy, GPU, or simulator is required.

Important distinction:
- across-state cosine: directions from different paired initial states;
- within-episode temporal cosine is NOT available from static captures;
- per-patch statistics retain the patch axis instead of treating feature
  channels as tokens.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def nearest_stats(query, reference, source_mean, source_scale):
    q = (query - source_mean) / source_scale
    r = (reference - source_mean) / source_scale
    d = np.linalg.norm(q[:, None, :] - r[None, :, :], axis=-1)
    mins = d.min(axis=1)
    return {"median": float(np.median(mins)), "p90": float(np.percentile(mins, 90)),
            "max": float(mins.max()), "mean": float(mins.mean()), "n": int(len(mins))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fit-start", type=int, default=10)
    ap.add_argument("--fit-count", type=int, default=10)
    args = ap.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    z = np.load(args.capture / "features.npz")
    rows = [r for r in manifest["records"] if r.get("compare_pass")]
    valid = [int(r["index"]) for r in rows]
    row = {int(r["index"]): int(r["feature_row"]) for r in rows}
    dev = [int(i) for i in manifest["dev_indices"] if int(i) in row]
    held = [int(i) for i in manifest["heldout_indices"] if int(i) in row]
    fit_ids = dev[args.fit_start:args.fit_start + args.fit_count]
    if len(fit_ids) < 2:
        raise RuntimeError("not enough source fit states")
    remaining_dev = [i for i in dev if i not in set(fit_ids)]
    eval_ids = remaining_dev + held
    state8 = z["state8"]
    source_x = state8[[row[i] for i in fit_ids]]
    source_mean = source_x.mean(axis=0)
    source_scale = source_x.std(axis=0); source_scale[source_scale < 1e-8] = 1.0
    out = {"capture": str(args.capture), "fit_ids": fit_ids,
           "remaining_dev": remaining_dev, "heldout_ids": held,
           "views": {}, "source_coverage": {}}
    for view in manifest["policy_views"]:
        clean = z[f"clean_{view}"]; shift = z[f"shift_{view}"]
        # [N, P, D] with P=patch tokens, D=feature channels.
        diff = shift - clean
        per_state = diff.mean(axis=1)                  # [N,D] for direction
        # Patch-token statistics: retain P axis.
        token_mean = diff.mean(axis=0)                 # [P,D]
        token_energy = np.linalg.norm(token_mean, axis=-1)  # [P]
        token_var = diff.var(axis=0).mean(axis=-1)     # [P]
        token_std = np.sqrt(token_var)
        stable_score = token_energy / (token_std + 1e-6)
        # Across-state direction stability: explicitly NOT temporal.
        cosines = [cosine(per_state[i], per_state[j])
                   for i in range(len(valid)) for j in range(i + 1, len(valid))]
        out["views"][view] = {
            "patch_tokens": int(diff.shape[1]),
            "feature_dim": int(diff.shape[2]),
            "residual_norm_mean": float(np.linalg.norm(per_state, axis=1).mean()),
            "residual_norm_std": float(np.linalg.norm(per_state, axis=1).std()),
            "across_state_cosine_mean": float(np.mean(cosines)),
            "across_state_cosine_std": float(np.std(cosines)),
            "top10_patch_energy_fraction": float(np.sort(token_energy)[-10:].sum() /
                                                 (token_energy.sum() + 1e-12)),
            "top10_patch_stability_fraction": float(np.sort(stable_score)[-10:].sum() /
                                                    (stable_score.sum() + 1e-12)),
            "patch_energy_mean": float(token_energy.mean()),
            "patch_energy_max": float(token_energy.max()),
        }
    # Source coverage using the actual V5 fit set and source-fit normalization.
    loo = []
    for i, s in enumerate(fit_ids):
        others = [t for t in fit_ids if t != s]
        loo.append(float(np.linalg.norm(
            (state8[row[s]] - source_mean) / source_scale -
            (state8[[row[t] for t in others]] - source_mean) / source_scale,
            axis=-1).min()))
    out["source_coverage"] = {
        "fit_loo": {"median": float(np.median(loo)), "p90": float(np.percentile(loo, 90)),
                    "max": float(np.max(loo)), "n": len(loo)},
        "remaining_dev_to_fit": nearest_stats(
            state8[[row[i] for i in remaining_dev]], source_x, source_mean, source_scale)
        if remaining_dev else None,
        "heldout_to_fit": nearest_stats(
            state8[[row[i] for i in held]], source_x, source_mean, source_scale)
        if held else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
