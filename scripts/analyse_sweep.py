"""Per-dimension paired analysis of the unified (state-conditioned) method sweep."""

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


def main():
    base = load("/hy-tmp/validation/lpfull_*/**/*.jsonl")
    rows = []
    for tag in ("Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"):
        meth = load(f"/hy-tmp/validation/sweep_{tag}_*/*.jsonl")
        keys = sorted(set(base) & set(meth))
        n = len(keys)
        if n == 0:
            rows.append((tag, 0, 0, 0, 0, 1.0, 0, 0))
            continue
        b = np.array([base[k] for k in keys], float)
        m = np.array([meth[k] for k in keys], float)
        ob = int(((b == 1) & (m == 0)).sum())
        oa = int(((b == 0) & (m == 1)).sum())
        p = binomtest(ob, ob + oa, 0.5).pvalue if ob + oa else 1.0
        rows.append((tag, n, b.mean() * 100, m.mean() * 100, (m.mean() - b.mean()) * 100, p, ob, oa))
    # Holm across dimensions
    ordered = sorted(range(len(rows)), key=lambda i: rows[i][5])
    holm = {}
    prev = 0.0
    for rank, i in enumerate(ordered):
        val = min(1.0, rows[i][5] * (len(rows) - rank))
        prev = max(prev, val)
        holm[i] = prev
    print(f"{'dim':12s} {'n':>5s} {'base':>7s} {'method':>7s} {'delta':>7s} {'disc':>8s} {'p':>8s} {'p_holm':>8s}")
    for i, (tag, n, b, m, d, p, ob, oa) in enumerate(rows):
        print(f"{tag:12s} {n:5d} {b:6.1f}% {m:6.1f}% {d:+6.1f} {ob:3d}/{oa:<3d} {p:8.4f} {holm[i]:8.4f}")


if __name__ == "__main__":
    main()
