"""B1 screening on the 24-episode capture (bounded, exploratory).

Signals available pre-update: multi-view agreement, historical action continuity.
Future-visual prediction is delayed and is NOT evaluated here (predictor must be
frozen/trained per LOO fold; deferred). Oracle success-action is not estimable under
LOO and is excluded.

Outcome: dp12_proximity (H=12 steps) because geometric progress failed acceptance.
Evaluation: LOO task, baseline linear model without task-id, cluster bootstrap,
within-task permutation.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

CAP = Path("/hy-tmp/validation/b1_capture_v2")
OUT = Path("/hy-tmp/validation/b1_screen")


def load():
    cap = [json.loads(l) for l in open(CAP / "capture.jsonl") if l.strip()]
    prog = [json.loads(l) for l in open(CAP / "progress_v5.jsonl") if l.strip()]
    pidx = {(p["suite"], p["task_id"], p["episode"], p["replan"]): p for p in prog}
    rows = []
    for r in cap:
        if "replan" not in r:
            continue
        key = (r["suite"], r["task_id"], r["episode"], r["replan"])
        p = pidx.get(key)
        if p is None or p.get("dp12_proximity") is None:
            continue
        rows.append({"suite": r["suite"], "task_id": r["task_id"], "episode": r["episode"],
                     "domain": r["domain"], "replan": r["replan"], "steps": r["steps"],
                     "mv": -float(np.mean(r["view_pairwise_distances"])),
                     "hist": None if r["history_leftover_distance"] is None
                             else -float(r["history_leftover_distance"]),
                     "p_now": p["p_proximity"], "y": p["dp12_proximity"],
                     "act_mag": float(np.mean(np.abs(np.asarray(r["action_chunk_norm"]))))})
    return rows


def design(rows, use_signal=None):
    feat = []
    for r in rows:
        f = [1.0, r["steps"] / 100.0, r["p_now"], r["act_mag"],
             1.0 if r["domain"] == "Camera" else 0.0,
             1.0 if r["suite"] == "libero_spatial" else 0.0]
        if use_signal is not None:
            f.append(r[use_signal])
        feat.append(f)
    return np.asarray(feat)


def spearman(a, b):
    if len(a) < 3: return np.nan
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def loo(rows, signal):
    tasks = sorted({(r["suite"], r["task_id"]) for r in rows})
    per = {"base": [], "sig": []}
    for t in tasks:
        tr = [r for r in rows if (r["suite"], r["task_id"]) != t]
        te = [r for r in rows if (r["suite"], r["task_id"]) == t and r["y"] is not None]
        if len(tr) < 20 or len(te) < 5: continue
        ytr = np.array([r["y"] for r in tr])
        for name, sig in (("base", None), ("sig", signal)):
            A = design(tr, sig); coef = np.linalg.lstsq(A, ytr, rcond=None)[0]
            if signal is not None:
                if any(r[signal] is None for r in te):
                    per[name].append(np.nan); continue
            B = design(te, sig); pred = B @ coef
            per[name].append(spearman(pred, np.array([r["y"] for r in te])))
    return per


def boot_ci(vals, n=1000, seed=0):
    vals = np.asarray([v for v in vals if not np.isnan(v)])
    if len(vals) < 3: return None
    rng = np.random.default_rng(seed)
    m = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n)]
    return [float(np.percentile(m, 5)), float(np.percentile(m, 95))]


def perm_p(rows, signal, obs, n=500, seed=1):
    rng = np.random.default_rng(seed)
    tasks = sorted({(r["suite"], r["task_id"]) for r in rows})
    cnt = 0
    for _ in range(n):
        rows2 = [dict(r) for r in rows]
        for t in tasks:
            idx = [i for i, r in enumerate(rows2) if (r["suite"], r["task_id"]) == t and r[signal] is not None]
            vals = [rows2[i][signal] for i in idx]
            rng.shuffle(vals)
            for i, v in zip(idx, vals): rows2[i][signal] = v
        m = np.nanmean(loo(rows2, signal)["sig"])
        if m >= obs: cnt += 1
    return float((cnt + 1) / (n + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    print("samples", len(rows), "with hist", sum(1 for r in rows if r["hist"] is not None))
    out = {"n": len(rows), "signals": {}}
    for sig in ("mv", "hist"):
        if any(r[sig] is None for r in rows):
            sub = [r for r in rows if r[sig] is not None]
        else:
            sub = rows
        per = loo(sub, sig)
        base = float(np.nanmean(per["base"])); sigm = float(np.nanmean(per["sig"]))
        out["signals"][sig] = {"n": len(sub), "loo_base_spearman": base,
                               "loo_signal_spearman": sigm,
                               "delta": sigm - base,
                               "ci90": boot_ci(per["sig"]),
                               "perm_p": perm_p(sub, sig, sigm)}
        print(sig, out["signals"][sig])
    (OUT / "screen.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
