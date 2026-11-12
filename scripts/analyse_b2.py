"""B2 analysis: paired update-vs-control counterfactual effects and signal predictiveness."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--src", default="/hy-tmp/validation/b2v3/b2_v3.jsonl")
_ap.add_argument("--out", default="/hy-tmp/validation/b2_analysis")
_args = _ap.parse_args()
SRC = Path(_args.src)
OUT = Path(_args.out)


def auc(score, labels):
    pos = [s for s, y in zip(score, labels) if y == 1]
    neg = [s for s, y in zip(score, labels) if y == 0]
    if not pos or not neg: return None
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(w / (len(pos) * len(neg)))


def spear(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)
    except Exception:
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])


DELTA = 0.01


def classify(tau):
    if tau > DELTA: return "beneficial"
    if tau < -DELTA: return "harmful"
    return "unidentifiable"


def cluster_boot(vals, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted(set(groups))
    by = {k: [v for v, g in zip(vals, groups) if g == k] for k in keys}
    out = []
    for _ in range(n):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        vals2 = [v for k in pick for v in by[k]]
        out.append(float(np.mean(vals2)))
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))] if out else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    pairs = {}
    for r in rows:
        key = (r["domain"], r["suite"], r["task_id"], r["episode"], r["branch"], r["candidate"])
        pairs.setdefault(key, {})[r["mode"]] = r
    recs = []
    for key, d in pairs.items():
        if "control" not in d or "update" not in d: continue
        c, u = d["control"], d["update"]
        recs.append({"domain": key[0], "suite": key[1], "task_id": key[2], "episode": key[3],
                     "branch": key[4], "candidate": key[5],
                     "dp_control": c["dp"], "dp_update": u["dp"], "tau": u["dp"] - c["dp"],
                     "success_control": c["success"], "success_update": u["success"],
                     "action_change": u["action_change_vs_control"],
                     "dbeta_norm": u["dbeta_norm"],
                     "mv": c.get("mv_signal"), "hist": c.get("hist_signal")})
    res = {"n_pairs": len(recs), "candidates": {}}
    for cand in sorted({r["candidate"] for r in recs}):
        sel = [r for r in recs if r["candidate"] == cand]
        taus = np.array([r["tau"] for r in sel])
        res["candidates"][cand] = {
            "n": len(sel), "tau_mean": float(taus.mean()), "tau_median": float(np.median(taus)),
            "tau_std": float(taus.std()), "tau_ci90": cluster_boot(
                [r["tau"] for r in sel], [(r["suite"], r["task_id"]) for r in sel]),
            "class_counts": {k: int(sum(1 for r in sel if classify(r["tau"]) == k))
                             for k in ("beneficial", "unidentifiable", "harmful")},
            "per_domain": {d: float(np.mean([r["tau"] for r in sel if r["domain"] == d]))
                           for d in sorted({r["domain"] for r in sel})},
            "frac_tau_pos": float((taus > 0).mean()),
            "dp_control_mean": float(np.mean([r["dp_control"] for r in sel])),
            "dp_update_mean": float(np.mean([r["dp_update"] for r in sel])),
            "success_control": float(np.mean([r["success_control"] for r in sel])),
            "success_update": float(np.mean([r["success_update"] for r in sel])),
            "action_change_mean": float(np.mean([r["action_change"] for r in sel])),
            "dbeta_norm_mean": float(np.mean([r["dbeta_norm"] for r in sel])),
            "spearman_tau_action": float(spear(taus, [r["action_change"] for r in sel])),
            "spearman_tau_dbeta": float(spear(taus, [r["dbeta_norm"] for r in sel])),
        }
    # signal prediction of tau>0, within task, per candidate
    res["signal_auc"] = {}
    for cand in sorted({r["candidate"] for r in recs}):
        sel = [r for r in recs if r["candidate"] == cand]
        tasks = sorted({(r["suite"], r["task_id"]) for r in sel})
        out = {}
        for sig in ("mv", "hist"):
            vals = []
            for t in tasks:
                ss = [r for r in sel if (r["suite"], r["task_id"]) == t and r[sig] is not None]
                if len(ss) < 3 or len({r["tau"] > 0 for r in ss}) < 2: continue
                vals.append(auc([r[sig] for r in ss], [int(r["tau"] > 0) for r in ss]))
            per_task = {t: v for t in tasks for v in [None]}
            # cluster bootstrap over available task AUCs
            boots = []
            if vals:
                rng = np.random.default_rng(0)
                for _ in range(1000):
                    pick = [vals[i] for i in rng.integers(0, len(vals), len(vals))]
                    boots.append(float(np.mean(pick)))
            out[sig] = {"n_tasks": len(vals), "mean_auc": float(np.mean(vals)) if vals else None,
                        "ci90": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))] if boots else None}
        res["signal_auc"][cand] = out
    (OUT / "b2_analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
