"""Offline monitor metrics, including first-alarm lead time per episode."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dacsc.failure_monitor import CausalFailureMonitor


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg): return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fpr", type=float, default=0.05)
    ap.add_argument("--source", type=Path, default=None,
                    help="source windows used only to fix the threshold")
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True); ck = torch.load(args.checkpoint, map_location="cpu")
    model = CausalFailureMonitor(window_size=ck["window"], hidden_dim=ck["hidden"],
                                 use_visual=ck.get("use_visual", True)); model.load_state_dict(ck["model"]); model.eval()
    mu = {k: np.asarray(v[0]) for k, v in ck["normalization"].items()}
    sd = {k: np.asarray(v[1]) for k, v in ck["normalization"].items()}
    s = ((d["state"] - mu["state"]) / sd["state"]).astype(np.float32)
    a = ((d["action"] - mu["action"]) / sd["action"]).astype(np.float32)
    v = ((d["visual_delta"] - mu["visual_delta"]) / sd["visual_delta"]).astype(np.float32)
    with torch.no_grad():
        p = model(torch.from_numpy(s), torch.from_numpy(a), torch.from_numpy(v)).numpy()
    y = d["y"]
    if args.source is not None:
        sd = np.load(args.source, allow_pickle=True)
        sd_s = ((sd["state"] - mu["state"]) / sd["state"]).astype(np.float32)
        sd_a = ((sd["action"] - mu["action"]) / sd["action"]).astype(np.float32)
        sd_v = ((sd["visual_delta"] - mu["visual_delta"]) / sd["visual_delta"]).astype(np.float32)
        with torch.no_grad():
            ps = model(torch.from_numpy(sd_s), torch.from_numpy(sd_a), torch.from_numpy(sd_v)).numpy()
        val_groups = set(ck.get("val_groups", []))
        vm = np.array([g in val_groups for g in sd["group"]])
        src_y = sd["y"][vm]
        src_p = ps[vm]
        neg = np.sort(src_p[src_y == 0])
        if len(neg) == 0: raise ValueError("source has no validation negatives")
        thr = float(np.quantile(neg, 1 - args.fpr))
        threshold_source = str(args.source)
    else:
        neg = np.sort(p[y == 0]); thr = float(np.quantile(neg, 1 - args.fpr)); threshold_source = None
    thr = float(thr)
    # per episode first alarm
    ep = d["episode"]; t = d["time"]; rem = d["remaining"]
    lead = []; false_alarm = 0
    for e in sorted(set(ep.tolist())):
        idx = np.where(ep == e)[0]
        fail = bool(y[idx].max() > 0.5)
        alarm = idx[p[idx] >= thr]
        if fail and len(alarm):
            lead.append(int(rem[alarm[0]]))
        elif not fail and len(alarm):
            false_alarm += 1
    report = {"auc": auc(y, p), "n_windows": len(y), "threshold": thr,
              "threshold_source": threshold_source,
              "recall_at_fpr": float(np.mean(p[y == 1] >= thr)) if np.any(y == 1) else None,
              "n_failure_episodes": int(len(set(ep[y > 0.5].tolist()))),
              "n_success_episodes": int(len(set(ep[y < 0.5].tolist()))),
              "first_alarm_remaining_steps": lead,
              "median_lead_steps": float(np.median(lead)) if lead else None,
              "false_alarm_episodes": false_alarm}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
