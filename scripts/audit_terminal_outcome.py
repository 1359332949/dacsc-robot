"""Gate 1: episode-level terminal outcome predictability audit on B1 capture.

One feature vector per episode at a fixed replan (default 8), label = final success.
Compares metadata-only, current-state, and transition-feature models with
leave-one-task-out logistic regression; task-cluster bootstrap CI is exploratory.
"""
from __future__ import annotations
import json, argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

CAP = Path("/tmp/b1v2")


def auc(score, label):
    pos = [s for s, y in zip(score, label) if y]
    neg = [s for s, y in zip(score, label) if not y]
    if not pos or not neg: return None
    return float(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)))


def std_fit(X):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0
    return mu, sd


def logistic_fit(X, y, l2=1.0, steps=2000, lr=0.1):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(steps):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = np.concatenate([X.T @ (p - y) / n + l2 * w / n, [np.mean(p - y)]])
        w -= lr * g[:-1]; b -= lr * g[-1]
    return w, b


def predict(X, w, b):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))


def loo_task(X, y, tasks):
    pred = np.zeros(len(y))
    for t in sorted(set(tasks)):
        te = np.array([x == t for x in tasks]); tr = ~te
        mu, sd = std_fit(X[tr]); w, b = logistic_fit((X[tr] - mu) / sd, y[tr])
        pred[te] = predict((X[te] - mu) / sd, w, b)
    return pred


def cluster_boot(pred, y, tasks, n=500, seed=0):
    rng = np.random.default_rng(seed); keys = sorted(set(tasks)); out = []
    for _ in range(n):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        ps = []; ys = []
        for k in pick:
            idx = np.where(np.array(tasks) == k)[0]
            ps.extend(pred[idx]); ys.extend(y[idx])
        a = auc(ps, ys)
        if a is not None: out.append(a)
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))] if out else None


def target_for(suite, instr):
    if suite == "libero_spatial": return "akita_black_bowl_1_main", "plate_1_main"
    for o in ("alphabet_soup_1_main", "salad_dressing_1_main", "butter_1_main"):
        if o.split("_")[0] in instr.lower(): return o, "basket_1_main"
    return "alphabet_soup_1_main", "basket_1_main"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replan", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("/tmp/terminal_outcome_audit.json"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(CAP / "capture.jsonl") if l.strip()]
    reps = [r for r in rows if "replan" in r]
    sums = [r["_episode_summary"] for r in rows if "_episode_summary" in r]
    success = {(s["suite"], s["task_id"], s["episode"]): bool(s["success"]) for s in sums}
    by = defaultdict(list)
    for r in reps: by[(r["suite"], r["task_id"], r["episode"])].append(r)
    feats = {"A": [], "B": [], "C": []}; labels = []; tasks = []
    for key, rs in sorted(by.items()):
        suite, tid, ep = key
        rs.sort(key=lambda x: x["replan"])
        idx = min(args.replan, len(rs) - 1)
        r = rs[idx]; p = rs[max(0, idx - 1)]
        names = r["body_names"]; bp = r["body_pos"]
        obj, tgt = target_for(suite, r["instruction"])
        oi, ti = names.index(obj), names.index(tgt)
        d = float(np.linalg.norm(np.array(bp[obj]) - np.array(bp[tgt])))
        eef = names.index("gripper0_eef")
        e = float(np.linalg.norm(np.array(bp[names[eef]]) - np.array(bp[obj])))
        d0 = float(np.linalg.norm(np.array(rs[0]["body_pos"][obj]) - np.array(rs[0]["body_pos"][tgt])))
        lift = float(np.array(bp[obj])[2] - np.array(rs[0]["body_pos"][obj])[2])
        meta = [1.0 if r["domain"] == "Camera" else 0.0,
                1.0 if suite == "libero_spatial" else 0.0, d0, e]
        state = list(np.asarray(r["state"], dtype=float).reshape(-1)) + [d / max(d0, 1e-9), e, lift]
        act = np.asarray(r["action_chunk_ctrl"], dtype=float)[:6]
        dp = np.asarray(p["state"], dtype=float).reshape(-1)
        ds = np.asarray(r["state"], dtype=float).reshape(-1) - dp[:len(np.asarray(r["state"]).reshape(-1))]
        trans = list(np.mean(np.abs(act), axis=0)) + list(np.std(act, axis=0)) + list(ds[:8]) + [
            d / max(d0, 1e-9), e, lift]
        feats["A"].append(meta); feats["B"].append(meta + state); feats["C"].append(meta + state + trans)
        labels.append(success[key]); tasks.append(f"{suite}|{tid}")
    y = np.array(labels, dtype=float)
    res = {"replan": args.replan, "n_episodes": len(y), "n_success": int(y.sum()),
           "n_tasks": len(set(tasks)), "models": {}}
    for name in ("A", "B", "C"):
        X = np.array(feats[name], dtype=float)
        pred = loo_task(X, y, tasks)
        res["models"][name] = {"n_features": X.shape[1], "loo_auc": auc(pred, y),
                               "cluster_boot_90": cluster_boot(pred, y, tasks),
                               "pred": [float(v) for v in pred]}
    res["delta_auc_B_minus_A"] = (res["models"]["B"]["loo_auc"] or 0) - (res["models"]["A"]["loo_auc"] or 0)
    res["delta_auc_C_minus_B"] = (res["models"]["C"]["loo_auc"] or 0) - (res["models"]["B"]["loo_auc"] or 0)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "models"}, indent=2))
    for name in ("A", "B", "C"):
        print(name, "features", res["models"][name]["n_features"],
              "AUC", res["models"][name]["loo_auc"],
              "CI", res["models"][name]["cluster_boot_90"])


if __name__ == "__main__":
    main()
