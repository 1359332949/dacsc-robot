"""Evaluate an ensemble monitor with source-calibrated threshold and persistence."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dacsc.monitor_ensemble import MonitorEnsemble


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg): return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def first_alarm(above, k):
    run = 0
    for j, f in enumerate(above):
        run = run + 1 if f else 0
        if run >= k: return j
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-val", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fpr", type=float, default=0.05)
    args = ap.parse_args()
    ens = MonitorEnsemble(args.artifact)
    sv = np.load(args.source_val, allow_pickle=True); t = np.load(args.target, allow_pickle=True)
    psv = ens.predict(sv["state"], sv["action"], sv["visual_delta"])
    pt = ens.predict(t["state"], t["action"], t["visual_delta"])
    neg = np.sort(psv[sv["y"] == 0]); thr = float(np.quantile(neg, 1 - args.fpr))
    k = int(ens.artifact["persistence"])
    rows = []
    for e in sorted(set(t["episode"].tolist())):
        idx = np.where(t["episode"] == e)[0]; idx = idx[np.argsort(t["time"][idx])]
        fail = bool(t["y"][idx].max() > 0.5)
        j = first_alarm(pt[idx] >= thr, k)
        alarm = j is not None
        lead = None
        if alarm:
            ev = int(t["failure_event_step"][idx][j]) if "failure_event_step" in t else -1
            lead = max(0, ev - (int(t["time"][idx][j]) + 1)) if ev >= 0 else int(t["remaining"][idx][j])
        rows.append({"episode": e, "state": str(t["group"][idx[0]]), "domain": str(t["domain"][idx[0]]),
                     "failure": fail, "alarm": alarm, "alarm_step": int(t["time"][idx][j]) if alarm else None,
                     "lead": lead, "max_score": float(pt[idx].max())})
    false = sum(1 for r in rows if r["alarm"] and not r["failure"])
    det = sum(1 for r in rows if r["alarm"] and r["failure"])
    nf = sum(1 for r in rows if r["failure"]); ns = len(rows) - nf
    leads = [r["lead"] for r in rows if r["alarm"] and r["failure"] and r["lead"] is not None]
    report = {"auc": auc(t["y"], pt), "threshold": thr, "persistence": k,
              "failure_detected": det, "n_failure_episodes": nf,
              "false_alarm_episodes": false, "n_success_episodes": ns,
              "median_lead": float(np.median(leads)) if leads else None,
              "per_episode": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "per_episode"}, indent=2))


if __name__ == "__main__":
    main()
