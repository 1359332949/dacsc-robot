"""B2 v3 analysis with correct success accounting and per-candidate/per-task summaries."""
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
    if not vals: return None
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


def _succ(r, side):
    return bool(r.get(f"success_{side}", False))


def _success_counts(rows):
    """Complete rows are neither-success by definition; incomplete rows count only if flags exist."""
    out = {"control_success": 0, "update_success": 0, "update_only": 0,
           "control_only": 0, "both": 0, "neither": 0, "not_observed": 0}
    for r in rows:
        complete = "tau_prox" in r and "censor" not in r
        has_flags = "success_control" in r or "success_update" in r
        if complete:
            out["neither"] += 1; continue
        if not has_flags:
            out["not_observed"] += 1; continue
        c, u = _succ(r, "control"), _succ(r, "update")
        out["control_success"] += int(c); out["update_success"] += int(u)
        if u and not c: out["update_only"] += 1
        elif c and not u: out["control_only"] += 1
        elif c and u: out["both"] += 1
        else: out["neither"] += 1
    return out


def summarise(rows):
    res = {"n_records": len(rows), "stage_skips": sum(1 for r in rows if r.get("censor") == "stage_or_success"),
           "candidates": {}}
    res["success_outcomes"] = _success_counts(rows)
    for cand in sorted({r["candidate"] for r in rows if "candidate" in r}):
        sel = [r for r in rows if r.get("candidate") == cand]
        comp = [r for r in sel if "tau_prox" in r and "censor" not in r]
        cens = [r for r in sel if "censor" in r]
        reasons = {}
        for r in cens:
            reasons[r["censor"]] = reasons.get(r["censor"], 0) + 1
        taus = [r["tau_prox"] for r in comp]
        groups = [(r["suite"], r["task_id"]) for r in comp]
        per_task = {}
        for r in comp:
            k = f"{r['suite']}|{r['task_id']}"
            per_task.setdefault(k, []).append(r["tau_prox"])
        res["candidates"][cand] = {
            "n_records": len(sel), "n_complete": len(comp), "n_censored": len(cens),
            "censor_reasons": reasons,
            "success_outcomes": _success_counts(sel),
            "tau_mean": float(np.mean(taus)) if taus else None,
            "tau_median": float(np.median(taus)) if taus else None,
            "tau_ci90": ci90(taus, groups) if taus else None,
            "class_counts": {k: sum(1 for t in taus if classify(t) == k)
                             for k in ("beneficial", "unidentifiable", "harmful")} if taus else None,
            "per_domain": {d: float(np.mean([r["tau_prox"] for r in comp if r["domain"] == d]))
                           for d in sorted({r["domain"] for r in comp})},
            "per_suite": {s: float(np.mean([r["tau_prox"] for r in comp if r["suite"] == s]))
                          for s in sorted({r["suite"] for r in comp})},
            "per_task": {k: {"n": len(v), "tau_mean": float(np.mean(v))} for k, v in per_task.items()},
            "spearman_tau_action": spear(taus, [r["action_change"] for r in comp]) if taus else None,
            "mean_action_change": float(np.mean([r["action_change"] for r in comp])) if comp else None,
            "mean_applied_norm": float(np.mean([r.get("applied_norm") or 0.0 for r in comp])) if comp else None,
            "mean_alpha": float(np.mean([r["alpha"] for r in comp if r.get("alpha") is not None])) if any(r.get("alpha") is not None for r in comp) else None,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/hy-tmp/validation/b2v3/b2_v3.jsonl")
    ap.add_argument("--out", default="/hy-tmp/validation/b2v3_analysis/analysis.json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.src) if l.strip()]
    res = summarise(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
