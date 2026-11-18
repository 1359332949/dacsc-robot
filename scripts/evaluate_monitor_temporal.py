"""Evaluate temporal alarm persistence for the failure monitor.

An episode alarms only after K consecutive windows exceed a source-fixed
threshold. This is post-processing, not threshold tuning on target test.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dacsc.failure_monitor import CausalFailureMonitor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-val", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--persistence", default="1,3,5,10")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = CausalFailureMonitor(window_size=ck["window"], hidden_dim=ck["hidden"],
                                 use_visual=ck.get("use_visual", True)); model.load_state_dict(ck["model"]); model.eval()
    mu = {k: np.asarray(v[0]) for k, v in ck["normalization"].items()}
    sd = {k: np.asarray(v[1]) for k, v in ck["normalization"].items()}
    def prep(d):
        return tuple(((d[k] - mu[k]) / sd[k]).astype(np.float32) for k in ("state", "action", "visual_delta"))
    def pred(d):
        s, a, v = prep(d)
        with torch.no_grad(): return model(torch.from_numpy(s), torch.from_numpy(a), torch.from_numpy(v)).numpy()
    sv = np.load(args.source_val, allow_pickle=True); t = np.load(args.target, allow_pickle=True)
    pv = pred(sv); pt = pred(t)
    neg = np.sort(pv[sv["y"] == 0]); thr = float(np.quantile(neg, 1 - args.fpr))
    rows = []
    for k in [int(x) for x in args.persistence.split(",")]:
        lead = []; false = 0; det = 0
        for e in sorted(set(t["episode"].tolist())):
            idx = np.where(t["episode"] == e)[0]
            idx = idx[np.argsort(t["time"][idx])]
            fail = bool(t["y"][idx].max() > 0.5)
            above = pt[idx] >= thr
            run = 0; alarm = None
            for j, flag in enumerate(above):
                run = run + 1 if flag else 0
                if run >= k:
                    alarm = j; break
            if alarm is not None:
                if fail:
                    det += 1
                    ev = int(t["failure_event_step"][idx][alarm]) if "failure_event_step" in t else -1
                    if ev >= 0:
                        lead.append(max(0, ev - int(t["time"][idx][alarm])))
                    else:
                        lead.append(int(t["remaining"][idx][alarm]))
                else:
                    false += 1
        nfail = len(set(t["episode"][t["y"] > 0.5].tolist()))
        nsucc = len(set(t["episode"][t["y"] < 0.5].tolist()))
        # same persistence on source validation, for selecting k without target
        sdet = 0; sfalse = 0; slead = []
        for e in sorted(set(sv["episode"].tolist())):
            idx = np.where(sv["episode"] == e)[0]
            idx = idx[np.argsort(sv["time"][idx])]
            fail = bool(sv["y"][idx].max() > 0.5)
            above = pv[idx] >= thr
            run = 0; alarm = None
            for j, flag in enumerate(above):
                run = run + 1 if flag else 0
                if run >= k: alarm = j; break
            if alarm is not None:
                if fail: sdet += 1; slead.append(int(sv["remaining"][idx][alarm]))
                else: sfalse += 1
        rows.append({"k": k, "threshold": thr,
                     "source_failure_detected": sdet,
                     "source_false_alarm_episodes": sfalse,
                     "source_median_lead_steps": float(np.median(slead)) if slead else None,
                     "target_failure_detected": det, "n_failure_episodes": nfail,
                     "target_false_alarm_episodes": false, "n_success_episodes": nsucc,
                     "target_median_lead_steps": float(np.median(lead)) if lead else None})
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
