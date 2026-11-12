"""B2 v3 analysis for the corrected schema (one record per candidate x episode x branch)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

DELTA = 0.01


def classify(tau):
    if tau > DELTA: return "beneficial"
    if tau < -DELTA: return "harmful"
    return "unidentifiable"


def ci90(vals, groups, n=1000, seed=0):
    keys = sorted(set(groups)); by = {k: [v for v, g in zip(vals, groups) if g == k] for k in keys}
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        out.append(float(np.mean([v for k in pick for v in by[k]])))
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))]


def spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]): return None
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)
    except Exception:
        return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/hy-tmp/validation/b2v3/b2_v3.jsonl")
    ap.add_argument("--out", default="/hy-tmp/validation/b2v3_analysis/analysis.json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.src) if l.strip()]
    complete = [r for r in rows if "tau_prox" in r and "censor" not in r]
    censored = [r for r in rows if "censor" in r]
    def succ(r, side):
        return bool(r.get(f"success_{side}", False))
    succ_rows = [r for r in rows if "success_control" in r or "success_update" in r]
    res = {"n_records": len(rows), "n_complete": len(complete), "n_censored": len(censored),
           "censor_reasons": {},
           "success_outcomes": {
               "control_success": sum(1 for r in succ_rows if succ(r, "control")),
               "update_success": sum(1 for r in succ_rows if succ(r, "update")),
               "update_only": sum(1 for r in succ_rows if succ(r, "update") and not succ(r, "control")),
               "control_only": sum(1 for r in succ_rows if succ(r, "control") and not succ(r, "update")),
               "both": sum(1 for r in succ_rows if succ(r, "control") and succ(r, "update")),
               "neither": sum(1 for r in succ_rows if not succ(r, "control") and not succ(r, "update")),
           },
           "candidates": {}}
    for r in censored:
        res["censor_reasons"][r["censor"]] = res["censor_reasons"].get(r["censor"], 0) + 1
    for cand in sorted({r["candidate"] for r in complete}):
        sel = [r for r in complete if r["candidate"] == cand]
        taus = [r["tau_prox"] for r in sel]
        groups = [(r["suite"], r["task_id"]) for r in sel]
        res["candidates"][cand] = {
            "n": len(sel), "tau_mean": float(np.mean(taus)), "tau_median": float(np.median(taus)),
            "tau_ci90": ci90(taus, groups),
            "class_counts": {k: sum(1 for t in taus if classify(t) == k)
                             for k in ("beneficial", "unidentifiable", "harmful")},
            "per_domain": {d: float(np.mean([r["tau_prox"] for r in sel if r["domain"] == d]))
                           for d in sorted({r["domain"] for r in sel})},
            "per_suite": {s: float(np.mean([r["tau_prox"] for r in sel if r["suite"] == s]))
                          for s in sorted({r["suite"] for r in sel})},
            "spearman_tau_action": spear(taus, [r["action_change"] for r in sel]),
            "mean_action_change": float(np.mean([r["action_change"] for r in sel])),
            "mean_effective_norm": float(np.mean([r["effective_norm"] for r in sel])),
            "mean_clip_ratio": float(np.mean([r["clip_ratio"] for r in sel])),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
