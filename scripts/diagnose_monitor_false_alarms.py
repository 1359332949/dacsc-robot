"""Per-episode false-alarm diagnosis with a source-fixed threshold."""

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
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = CausalFailureMonitor(window_size=ck["window"], hidden_dim=ck["hidden"],
                                 use_visual=ck.get("use_visual", True))
    model.load_state_dict(ck["model"]); model.eval()
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
    false = []
    for e in sorted(set(t["episode"].tolist())):
        idx = np.where(t["episode"] == e)[0]
        fail = bool(t["y"][idx].max() > 0.5)
        alarm = idx[pt[idx] >= thr]
        if not fail and len(alarm):
            false.append({"episode": e, "state": str(t["group"][idx[0]]), "domain": str(t["domain"][idx[0]]),
                          "first_alarm_time": int(t["time"][alarm[0]]), "max_score": float(pt[idx].max()),
                          "n_windows": int(len(idx))})
    report = {"schema": "monitor_false_alarm_diag_v1", "threshold": thr,
              "n_target_windows": len(pt), "false_alarm_episodes": false,
              "false_alarm_count": len(false), "target_success_episodes": len(set(t["episode"][t["y"] < 0.5].tolist()))}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
