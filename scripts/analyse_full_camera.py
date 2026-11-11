"""Paired analysis of the full Camera dimension run against the official baseline."""

import glob
import json

import numpy as np
from scipy.stats import binomtest


def load(pattern: str) -> dict:
    out = {}
    for f in glob.glob(pattern, recursive=True):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                out[(r["suite"], r["task_id"], r["episode_id"])] = bool(r["success"])
    return out


def paired_stats(base: dict, meth: dict, keys, label: str) -> dict | None:
    n = len(keys)
    if n == 0:
        return None
    b = np.array([base[k] for k in keys], dtype=float)
    m = np.array([meth[k] for k in keys], dtype=float)
    ob = int(np.sum((b == 1) & (m == 0)))
    oa = int(np.sum((b == 0) & (m == 1)))
    p = binomtest(ob, ob + oa, 0.5).pvalue if ob + oa else 1.0
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        deltas.append((m[idx].mean() - b[idx].mean()) * 100)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "label": label,
        "n": n,
        "base": b.mean() * 100,
        "meth": m.mean() * 100,
        "delta": (m.mean() - b.mean()) * 100,
        "ci_lo": lo,
        "ci_hi": hi,
        "disc": ob,
        "disc2": oa,
        "p": p,
    }


def main() -> int:
    base = load("/hy-tmp/validation/lpfull_*/**/*.jsonl")
    meth = {}
    for s in ("spatial", "object", "goal", "10"):
        meth.update(load(f"/hy-tmp/validation/full_cam_{s}/*.jsonl"))
    keys = sorted(set(base) & set(meth))
    cam = json.load(open("/hy-tmp/validation/camera_full.json"))
    level = {}
    for s, items in cam.items():
        for t in items:
            level[(f"libero_{s}", t["task_id"], 0)] = t["level"]

    rows = [paired_stats(base, meth, keys, "ALL camera")]
    for s in ("spatial", "object", "goal", "10"):
        ks = [k for k in keys if k[0] == f"libero_{s}"]
        rows.append(paired_stats(base, meth, ks, f"  {s}"))
    for lv in (1, 2, 3, 4, 5):
        ks = [k for k in keys if level.get(k) == lv]
        rows.append(paired_stats(base, meth, ks, f"  L{lv}"))

    level_indices = [i for i, r in enumerate(rows) if r and r["label"].startswith("  L")]
    ordered = sorted(level_indices, key=lambda i: rows[i]["p"])
    holm = {}
    prev = 0.0
    for rank, i in enumerate(ordered):
        val = min(1.0, rows[i]["p"] * (len(ordered) - rank))
        prev = max(prev, val)
        holm[i] = prev

    for i, r in enumerate(rows):
        if not r:
            continue
        extra = f" p_holm={holm[i]:.4f}" if i in holm else ""
        print(
            f"{r['label']:16s} n={r['n']:4d} base={r['base']:5.1f}% method={r['meth']:5.1f}% "
            f"delta={r['delta']:+5.1f} [{r['ci_lo']:+5.1f},{r['ci_hi']:+5.1f}] "
            f"disc={r['disc']}/{r['disc2']} p={r['p']:.4f}{extra}"
        )
    print("total paired so far:", len(keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
