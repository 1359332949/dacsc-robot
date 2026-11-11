"""Interim paired analysis of the re-validation variants (corrected protocol)."""

import glob
import json

import numpy as np
from scipy.stats import binomtest


def load(pattern):
    out = {}
    for f in glob.glob(pattern, recursive=True):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                out[(r["suite"], r["task_id"], r["episode_id"])] = bool(r["success"])
    return out


def stats(base, meth, label):
    keys = sorted(set(base) & set(meth))
    n = len(keys)
    if n == 0:
        print(f"{label:16s} no pairs")
        return
    b = np.array([base[k] for k in keys], dtype=float)
    m = np.array([meth[k] for k in keys], dtype=float)
    ob = int(np.sum((b == 1) & (m == 0)))
    oa = int(np.sum((b == 0) & (m == 1)))
    p = binomtest(ob, ob + oa, 0.5).pvalue if ob + oa else 1.0
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        ds.append((m[idx].mean() - b[idx].mean()) * 100)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    print(
        f"{label:16s} n={n:4d} base={b.mean()*100:5.1f}% method={m.mean()*100:5.1f}% "
        f"delta={(m.mean()-b.mean())*100:+5.1f} [{lo:+5.1f},{hi:+5.1f}] disc={ob}/{oa} p={p:.4f}"
    )


base = load("/hy-tmp/validation/lpfull_*/**/*.jsonl")
for tag in ("rv_cam_topk", "rv_cam_aff", "rv_cam_aff_s5", "rv_cam_aff_a05",
            "rv_l3_topk", "rv_l3_aff"):
    stats(base, load(f"/hy-tmp/validation/{tag}/*.jsonl"), tag)
deb = load("/hy-tmp/validation/full_cam_spatial/*.jsonl")
stats(base, deb, "full_debias_spatial")
