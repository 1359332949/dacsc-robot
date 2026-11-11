"""Paired analysis with Newcombe hybrid-score CI, cluster bootstrap and TOST.

Usage:
  python scripts/analyse_matrix.py --base <base.jsonl> --method <method.jsonl> [--margin 0.05]
  python scripts/analyse_matrix.py --pairs <pairs.jsonl>   # rows with base_success/method_success
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
from scipy.stats import binomtest

try:
    from scipy.stats import norm
except Exception:  # pragma: no cover
    norm = None


def load_records(paths: list[str]) -> dict[tuple[int, int], bool]:
    out = {}
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[(r["task_id"], r["episode_id"])] = bool(r["success"])
    return out


def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_paired(a: int, b: int, c: int, d: int, z: float = 1.645) -> tuple[float, float]:
    """Newcombe (1998) method 10 hybrid score interval for p2-p1 (paired)."""
    n = a + b + c + d
    if n == 0:
        return -1.0, 1.0
    p1 = (a + b) / n
    p2 = (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = 0.0 if den == 0 else (a * d - b * c) / den
    lower = (p2 - p1) - math.sqrt(max(0.0, (p2 - l2) ** 2 + (u1 - p1) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1)))
    upper = (p2 - p1) + math.sqrt(max(0.0, (u2 - p2) ** 2 + (p1 - l1) ** 2 + 2 * phi * (u2 - p2) * (p1 - l1)))
    return lower, upper


def cluster_bootstrap(per_task: dict[int, list[tuple[bool, bool]]], n_boot: int = 5000, seed: int = 0,
                      level: float = 0.90) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    tasks = sorted(per_task)
    deltas = []
    for _ in range(n_boot):
        sampled = rng.choice(tasks, size=len(tasks), replace=True)
        base = np.array([b for t in sampled for b, _ in per_task[t]], float)
        meth = np.array([m for t in sampled for _, m in per_task[t]], float)
        if len(base):
            deltas.append(meth.mean() - base.mean())
    if not deltas:
        return 0.0, 0.0
    lo = float(np.percentile(deltas, (1 - level) / 2 * 100))
    hi = float(np.percentile(deltas, (1 + level) / 2 * 100))
    return lo, hi


def analyse(base: dict[tuple[int, int], bool], meth: dict[tuple[int, int], bool], margin: float | None = None) -> dict:
    keys = sorted(set(base) & set(meth))
    n = len(keys)
    a = sum(1 for k in keys if base[k] and meth[k])
    b = sum(1 for k in keys if base[k] and not meth[k])
    c = sum(1 for k in keys if not base[k] and meth[k])
    d = sum(1 for k in keys if not base[k] and not meth[k])
    delta = (c - b) / n if n else 0.0
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    lo, hi = newcombe_paired(a, b, c, d, z=1.645)
    per_task: dict[int, list[tuple[bool, bool]]] = {}
    for k in keys:
        per_task.setdefault(k[0], []).append((base[k], meth[k]))
    clo, chi = cluster_bootstrap(per_task, level=0.90)
    res = {
        "n": n, "base": (a + b) / n if n else 0.0, "method": (a + c) / n if n else 0.0,
        "n_base_only": len(set(base) - set(meth)), "n_method_only": len(set(meth) - set(base)),
        "n_tasks": len({k[0] for k in keys}),
        "delta": delta, "discordant": f"{b}/{c}", "mcnemar_p": p,
        "newcombe90": [lo, hi], "cluster90": [clo, chi],
    }
    if margin is not None:
        # two DIFFERENT operational equivalence judgements
        res["tost_paired_pass"] = (lo > -margin) and (hi < margin)
        res["tost_cluster_pass"] = (clo > -margin) and (chi < margin)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="*", default=None)
    ap.add_argument("--method", nargs="*", default=None)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if args.pairs:
        rows = [json.loads(l) for l in open(args.pairs) if l.strip()]
        base = {(r["task_id"], 0): bool(r["base_success"]) for r in rows}
        meth = {(r["task_id"], 0): bool(r["oracle_success"]) for r in rows}
    else:
        base = load_records(args.base)
        meth = load_records(args.method)
    res = analyse(base, meth, args.margin)
    print(args.label, json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
