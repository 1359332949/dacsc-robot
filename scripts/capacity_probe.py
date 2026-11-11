"""Can a shared low-rank MULTIPLICATIVE interface transform express the paired Oracle shift?

For each held-out task:
    fit M (D x D) on train tasks by ridge regression:  Z M ~= D
      Z = w_shift (D=768), D = w_clean - w_shift (= -delta, the additive correction)
    truncate M to rank r (SVD) and evaluate on the held-out task.

Comparison models:
    oracle_additive     : use D exactly                 (upper bound)
    global_mean_shift   : mean(D) over train tasks      (state-independent additive)
    step_conditioned    : linear in [1, step_norm]      (time-conditioned additive)
    lowrank_mul_r       : w_shift @ M_r                 (what CPIA can express)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

RANKS = (1, 2, 4, 8, 16, 32, 64)


def load_pairs(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    data = np.load(path)
    tasks: dict[int, tuple[list, list]] = {}
    for key in data.files:
        if key.startswith("shift_"):
            tid = int(key.split("_")[1])
            tasks.setdefault(tid, (None, None))
    out = {}
    for tid in tasks:
        shift = data[f"shift_{tid}"].astype(np.float64)
        clean = data[f"clean_{tid}"].astype(np.float64)
        n = min(len(shift), len(clean))
        out[tid] = (shift[:n], clean[:n] - shift[:n])
    return out


def fit_ridge(Z: np.ndarray, D: np.ndarray, lam: float = 1.0) -> np.ndarray:
    d = Z.shape[1]
    A = Z.T @ Z + lam * np.eye(d)
    return np.linalg.solve(A, Z.T @ D)


def rank_truncate(M: np.ndarray, r: int) -> np.ndarray:
    u, s, vt = np.linalg.svd(M, full_matrices=False)
    return (u[:, :r] * s[:r]) @ vt[:r]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    tasks = load_pairs(args.pairs)
    ids = sorted(tasks)
    print(f"tasks={len(ids)} samples={sum(len(tasks[t][0]) for t in ids)}")
    results = {name: [] for name in ("oracle_additive", "global_mean_shift", "step_conditioned",
                                     *[f"lowrank_mul_r{r}" for r in RANKS])}
    for held in ids:
        train = [t for t in ids if t != held]
        Ztr = np.concatenate([tasks[t][0] for t in train])
        Dtr = np.concatenate([tasks[t][1] for t in train])
        Zte, Dte = tasks[held]
        # oracle additive
        results["oracle_additive"].append(1.0)
        # global mean shift
        pred = np.broadcast_to(Dtr.mean(0), Dte.shape)
        results["global_mean_shift"].append(cos(pred, Dte))
        # step-conditioned additive: phi = [1, step_norm]; step_norm = index/15 capped at 1
        def phi(Z):
            idx = np.arange(len(Z))[:, None] / 15.0
            return np.concatenate([np.ones((len(Z), 1)), np.clip(idx, 0, 1)], axis=1)
        W = np.linalg.solve(phi(Ztr).T @ phi(Ztr) + 1.0 * np.eye(2), phi(Ztr).T @ Dtr)
        results["step_conditioned"].append(cos(phi(Zte) @ W, Dte))
        # shared multiplicative low-rank (CPIA hypothesis class)
        M = fit_ridge(Ztr, Dtr, args.lam)
        norm = float(np.linalg.norm(Dte))
        for r in RANKS:
            pred = Zte @ rank_truncate(M, r)
            results[f"lowrank_mul_r{r}"].append(cos(pred, Dte))
    print(f"{'model':22s} {'mean cosine':>12s} {'std':>8s}")
    summary = {}
    for name, vals in results.items():
        vals = np.asarray(vals)
        summary[name] = float(vals.mean())
        print(f"{name:22s} {vals.mean():12.3f} {vals.std():8.3f}")
    if args.out is not None:
        import json
        args.out.write_text(json.dumps({"summary": summary}, indent=2))
    return 0


def cos(pred: np.ndarray, target: np.ndarray) -> float:
    num = float(np.sum(pred * target))
    den = float(np.linalg.norm(pred) * np.linalg.norm(target))
    return num / den if den > 0 else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
