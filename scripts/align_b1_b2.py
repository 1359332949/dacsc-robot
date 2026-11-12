"""Offline B1-B2 alignment: verify same branch observation, then exploratory signal->utility."""
import json
from pathlib import Path
import numpy as np

B1 = Path("/hy-tmp/validation/b1_capture_v2/capture.jsonl")
B2 = Path("/hy-tmp/validation/b2v3_formal/b2_v3.jsonl")
OUT = Path("/hy-tmp/validation/b1_b2_alignment.json")


def spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 4 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)
    except Exception:
        return float(np.corrcoef(a, b)[0, 1])


def loo_spear(x, y, tasks):
    keys = sorted(set(tasks))
    if len(keys) < 3: return None
    pred = np.full(len(y), np.nan)
    for k in keys:
        te = np.array([t == k for t in tasks])
        tr = ~te
        if tr.sum() < 4 or te.sum() < 1: continue
        A = np.column_stack([np.ones(tr.sum()), x[tr]])
        coef = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred[te] = coef[0] + coef[1] * x[te]
    m = ~np.isnan(pred)
    return spear(pred[m], y[m]) if m.sum() >= 4 else None


def loo_mean_spear(y, tasks):
    keys = sorted(set(tasks)); pred = np.full(len(y), np.nan)
    for k in keys:
        te = np.array([t == k for t in tasks]); tr = ~te
        if tr.sum() == 0: continue
        pred[te] = float(np.mean(y[tr]))
    m = ~np.isnan(pred)
    return spear(pred[m], y[m]) if m.sum() >= 4 else None


def perm_p(x, y, tasks, obs, n=500, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted(set(tasks)); cnt = 0
    for _ in range(n):
        x2 = x.copy()
        for k in keys:
            idx = np.where(np.array(tasks) == k)[0]
            vals = x2[idx].copy(); rng.shuffle(vals); x2[idx] = vals
        s = loo_spear(x2, y, tasks)
        if s is not None and abs(s) >= abs(obs): cnt += 1
    return float((cnt + 1) / (n + 1))


def main():
    b1 = [json.loads(l) for l in open(B1) if l.strip()]
    b1 = [r for r in b1 if "replan" in r]
    idx = {(r["suite"], r["task_id"], r["episode"], r["replan"]): r for r in b1}
    b2 = [json.loads(l) for l in open(B2) if l.strip()]
    cand = [r for r in b2 if "candidate" in r and "tau_prox" in r and "censor" not in r]
    # align branch rows and verify geometry
    branch = {}
    align = []
    for r in cand:
        key = (r["suite"], r["task_id"], r["episode"], r["branch"])
        if key in branch: continue
        b = idx.get(key)
        if b is None: continue
        names = b["body_names"]; bp = b["body_pos"]
        obj, tgt = r["target_pair"]
        oi, ti = names.index(obj + "_main"), names.index(tgt + "_main")
        gi = names.index("gripper0_eef")
        contact = float(np.linalg.norm(np.array(bp[names[gi]]) - np.array(bp[names[oi]])))
        dbranch = float(np.linalg.norm(np.array(bp[names[oi]]) - np.array(bp[names[ti]])))
        hist = b.get("history_leftover_distance")
        mv = -float(np.mean(b["view_pairwise_distances"]))
        branch[key] = {"mv": mv, "hist": None if hist is None else -float(hist),
                       "tau_sens": None, "tau_proj": None, "tau_full": None, "tau_prod": None}
        align.append({"key": key, "contact_b1": contact, "contact_b2": r.get("target_contact"),
                      "dbranch_b1": dbranch, "dbranch_b2": r.get("d_branch_control")})
    for r in cand:
        key = (r["suite"], r["task_id"], r["episode"], r["branch"])
        if key not in branch: continue
        c = r["candidate"]
        if c == "raw_sens_100": branch[key]["tau_sens"] = r["tau_prox"]
        elif c == "raw_proj_100": branch[key]["tau_proj"] = r["tau_prox"]
        elif c == "raw_full_100": branch[key]["tau_full"] = r["tau_prox"]
        elif c == "prod_step_005": branch[key]["tau_prod"] = r["tau_prox"]
    contact_err = [abs(a["contact_b1"] - a["contact_b2"]) for a in align if a["contact_b2"] is not None]
    db_err = [abs(a["dbranch_b1"] - a["dbranch_b2"]) for a in align if a["dbranch_b2"] is not None]
    res = {"n_aligned_branches": len(align),
           "max_contact_abs_diff": max(contact_err) if contact_err else None,
           "max_dbranch_abs_diff": max(db_err) if db_err else None,
           "n_with_sens": 0, "signal_prediction": {}}
    rows = [v for v in branch.values()]
    for outcome in ("tau_sens", "tau_proj", "tau_full", "tau_sens_minus_proj"):
        y = []
        for v in rows:
            if outcome == "tau_sens_minus_proj":
                if v["tau_sens"] is not None and v["tau_proj"] is not None:
                    y.append((v["tau_sens"] - v["tau_proj"], v))
            elif v[outcome] is not None:
                y.append((v[outcome], v))
        if len(y) < 6: continue
        yy = np.array([a for a, _ in y]); tasks = ["|".join(map(str, (v.get("suite"), v.get("task_id")))) if False else "" for _, v in y]
        # rebuild tasks from key ordering
        keys = [k for k in branch if branch[k] in [v for _, v in y]]
        tasks = ["|".join(map(str, k[:2])) for k in keys]
        res["signal_prediction"][outcome] = {"n": len(yy), "baseline_loo_task_mean_spearman": loo_mean_spear(yy, tasks)}
        for sig in ("mv", "hist"):
            x = np.array([v[sig] if v[sig] is not None else np.nan for _, v in y])
            m = ~np.isnan(x)
            if m.sum() < 6: continue
            xs, ys, ts = x[m], yy[m], [tasks[i] for i in np.where(m)[0]]
            obs = loo_spear(xs, ys, ts)
            res["signal_prediction"][outcome][sig] = {
                "n": int(m.sum()), "overall_spearman": spear(xs, ys),
                "loo_task_spearman": obs,
                "perm_p": perm_p(xs, ys, ts, obs) if obs is not None else None,
                "baseline_loo_task_mean_spearman": None}
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
