"""V1.2 per-view scores from captured paired DINO features.

No cross-view averaging.  agentview is the target view; eye-in-hand is a control.
Definitions:
  s_pair_v(d) = mean_{state,patch} |F_clean - F_shift|
  s_stat_v(d) = std_state( mean_patch F_shift )   [primary]
  s_stat_alt_v(d) = mean_patch( std_state F_shift ) [sensitivity analysis]
Reliability: dev/held rank correlation, top-k overlap, repeated split-half,
state bootstrap.  s_aug is intentionally not computed here yet.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr




def pair_score(clean: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Per-dimension s_pair; input [N, P, D], output [D]. No view mixing."""
    if clean.shape != shift.shape or clean.ndim != 3:
        raise ValueError("clean/shift must have same [N,P,D] shape")
    return np.abs(clean.astype(np.float64) - shift.astype(np.float64)).mean(axis=(0, 1))


def stat_primary(shift: np.ndarray) -> np.ndarray:
    """std_state(mean_patch F), output [D]."""
    return shift.astype(np.float64).mean(axis=1).std(axis=0)


def stat_alt(shift: np.ndarray) -> np.ndarray:
    """mean_patch(std_state F), output [D]."""
    return shift.astype(np.float64).std(axis=0).mean(axis=0)


def valid_positions(manifest: dict) -> tuple[list[int], dict[int, int]]:
    valid = [r["index"] for r in manifest["records"] if r.get("compare_pass")]
    return valid, {state: j for j, state in enumerate(valid)}


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ia = set(np.argsort(-a)[:k].tolist()); ib = set(np.argsort(-b)[:k].tolist())
    return len(ia & ib) / float(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261116)
    args = ap.parse_args()
    z = np.load(args.dir / "features.npz")
    manifest = json.loads((args.dir / "manifest.json").read_text())
    views = manifest["policy_views"]
    valid = [r["index"] for r in manifest["records"] if r.get("compare_pass")]
    pos = {state: j for j, state in enumerate(valid)}
    dev_idx = [i for i in manifest["dev_indices"] if i in valid]
    held_idx = [i for i in manifest["heldout_indices"] if i in valid]
    dev_pos = [pos[i] for i in dev_idx]; held_pos = [pos[i] for i in held_idx]
    rng = np.random.RandomState(args.seed)

    def pair_for(pos_idx, view):
        return pair_score(z[f"clean_{view}"][pos_idx], z[f"shift_{view}"][pos_idx])

    def stat_for(pos_idx, view, alt=False):
        s = z[f"shift_{view}"][pos_idx]
        return stat_alt(s) if alt else stat_primary(s)

    out = {"num_states": len(valid), "dev": len(dev_pos), "heldout": len(held_pos),
           "skipped": len(manifest["records"]) - len(valid), "views": {},
           "reliability": {}, "bootstrap": {}}
    for view in views:
        dev_pair = pair_for(dev_pos, view); held_pair = pair_for(held_pos, view)
        all_pair = pair_for(list(range(len(valid))), view)
        stat_dev = stat_for(dev_pos, view); stat_held = stat_for(held_pos, view)
        stat_alt = stat_for(list(range(len(valid))), view, alt=True)
        # repeated split-half on dev
        rhos = []
        for _ in range(args.repeat):
            p = rng.permutation(len(dev_pos)); h = len(p) // 2
            a = pair_for([dev_pos[j] for j in p[:h]], view)
            b = pair_for([dev_pos[j] for j in p[h:]], view)
            rhos.append(float(spearmanr(a, b).statistic))
        rhos = np.asarray(rhos, dtype=float)
        # state bootstrap for s_pair mean/top-k frequency
        boot_means = []; top_freq = np.zeros(all_pair.shape[0], dtype=float)
        for _ in range(args.bootstrap):
            idx = rng.randint(0, len(valid), len(valid))
            sc = pair_for(idx, view)
            boot_means.append(float(sc.mean()))
            top_freq[np.argsort(-sc)[:args.k]] += 1.0
        top_freq /= args.bootstrap
        out["views"][view] = {
            "pair_dev": dev_pair.tolist(), "pair_heldout": held_pair.tolist(),
            "pair_all": all_pair.tolist(),
            "stat_dev": stat_dev.tolist(), "stat_heldout": stat_held.tolist(),
            "stat_alt_all": stat_alt.tolist(),
            "pair_mean_dev": float(dev_pair.mean()),
            "pair_mean_heldout": float(held_pair.mean()),
            "top_dims_all": np.argsort(-all_pair)[:args.k].tolist(),
        }
        out["reliability"][view] = {
            "dev_vs_held_spearman": float(spearmanr(dev_pair, held_pair).statistic),
            "dev_vs_held_topk_overlap": topk_overlap(dev_pair, held_pair, args.k),
            "repeated_split_half_median": float(np.nanmedian(rhos)),
            "repeated_split_half_q05": float(np.nanquantile(rhos, 0.05)),
            "repeated_split_half_q95": float(np.nanquantile(rhos, 0.95)),
        }
        out["bootstrap"][view] = {
            "pair_mean_ci95": [float(np.quantile(boot_means, 0.025)),
                               float(np.quantile(boot_means, 0.975))],
            "topk_frequency_max": float(top_freq.max()),
            "topk_dim_stability": top_freq.tolist(),
        }
    (args.dir / "scores_per_view.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(args.dir / "scores_per_view.npz",
                        **{f"pair_all_{v}": np.asarray(out["views"][v]["pair_all"]) for v in views},
                        **{f"stat_heldout_{v}": np.asarray(out["views"][v]["stat_heldout"]) for v in views})
    print(json.dumps({v: {
        "pair_dev_mean": out["views"][v]["pair_mean_dev"],
        "pair_heldout_mean": out["views"][v]["pair_mean_heldout"],
        "reliability": out["reliability"][v],
        "bootstrap_pair_mean_ci95": out["bootstrap"][v]["pair_mean_ci95"],
    } for v in views}, indent=2))


if __name__ == "__main__":
    main()
