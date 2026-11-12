from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--analysis", default="/hy-tmp/validation/p42_analysis/phase4_2_analysis.json")
_ap.add_argument("--out", default="/hy-tmp/validation/p42_figures")
_args = _ap.parse_args()
AN = Path(_args.analysis)
OUT = Path(_args.out)
OUT.mkdir(parents=True, exist_ok=True)
d = json.loads(AN.read_text())
rows = [r for r in d["rows"]]
doms = [r["domain"] for r in rows]
methods = [("frozen", "Frozen"), ("salt", "SALT"), ("sg_cpa", "SG-CPA"),
           ("sg_noproj", "SG-CPA\nno-proj"), ("sg_excl", "SG-CPA\nmedoid excl-id")]
fig, ax = plt.subplots(figsize=(9, 4.2))
w = 0.16
x = np.arange(len(doms))
for i, (m, label) in enumerate(methods):
    vals = [r.get(f"sr_{m}", 0) * 100 for r in rows]
    ax.bar(x + (i - 2) * w, vals, w, label=label)
ax.set_xticks(x); ax.set_xticklabels(doms)
ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 100)
ax.set_title("Multi-domain LIBERO-Plus: Frozen / SALT / SG-CPA variants (seeds 30-34)")
ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(); fig.savefig(OUT / "sr_by_domain.png", dpi=160); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.8))
for j, (key, label, color) in enumerate([
        ("sg_cpa_minus_salt_ci90", "SG-CPA - SALT", "#d62728"),
        ("sg_excl_minus_salt_ci90", "SG-CPA(excl-id) - SALT", "#2ca02c")]):
    xs, ys, lo, hi = [], [], [], []
    for i, r in enumerate(rows):
        ci = r.get(key)
        if ci is None:
            continue
        xs.append(i + (j - 0.5) * 0.18)
        ys.append(r["mcnemar_sg_cpa_vs_salt"]["pair_diff"] * 100 if key.startswith("sg_cpa_") else r["mcnemar_sg_excl_vs_salt"]["pair_diff"] * 100)
        lo.append(ys[-1] - ci[0] * 100); hi.append(ci[1] * 100 - ys[-1])
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=color, label=label, capsize=3)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xticks(range(len(rows))); ax.set_xticklabels(doms)
ax.set_ylabel("Success-rate difference (pp)")
ax.set_title("Paired difference vs SALT (task-cluster bootstrap 90% CI)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "delta_vs_salt.png", dpi=160); plt.close(fig)
print("saved", sorted(p.name for p in OUT.glob("*.png")))
