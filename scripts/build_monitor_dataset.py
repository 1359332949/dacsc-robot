"""Build sliding-window monitor datasets from per-step trajectory JSONs.

Deployment inputs only: post-step state8, executed (clipped) action, and
visual-feature delta. Labels are episode-level failure for training/evaluation
only; they are not monitor inputs.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def load_episode(path: Path) -> dict:
    d = json.loads(path.read_text())
    rows = d.get("rows", [])
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    state = np.asarray([r["state8"] for r in rows], dtype=np.float32)
    action = np.asarray([r["clipped_action"] for r in rows], dtype=np.float32)
    visual = np.asarray([r["visual_mean"] for r in rows], dtype=np.float32)
    if state.shape[1] != 8 or action.shape[1] != 7 or visual.shape[1] != 768:
        raise ValueError(f"bad feature shape in {path}: {state.shape} {action.shape} {visual.shape}")
    delta = np.zeros_like(visual)
    if len(visual) > 1:
        delta[1:] = visual[1:] - visual[:-1]
    return {"state": state, "action": action, "visual_delta": delta,
            "failure": int(not d["success"]), "success": bool(d["success"]),
            "steps": len(rows), "state_id": d["state_id"], "domain": d["domain"],
            "seed": d["seed"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-dir", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    Xs, As, Vs, Ys, remaining, groups, domains = [], [], [], [], [], [], []
    n_ep = 0
    for d in args.trajectory_dir:
        for p in sorted(d.glob("s*_d*_seed*.json")):
            e = load_episode(p); n_ep += 1
            n = len(e["state"])
            for t in range(args.window - 1, n, args.stride):
                sl = slice(t - args.window + 1, t + 1)
                Xs.append(e["state"][sl]); As.append(e["action"][sl]); Vs.append(e["visual_delta"][sl])
                Ys.append(e["failure"]); remaining.append(n - 1 - t)
                groups.append(str(e['state_id']))
                domains.append(e["domain"])
    if not Xs:
        raise ValueError("no windows built")
    y = np.asarray(Ys, dtype=np.float32)
    if len(set(y.tolist())) < 2:
        raise ValueError("dataset has only one class; collect failures and successes")
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, state=np.stack(Xs), action=np.stack(As),
                        visual_delta=np.stack(Vs), y=y,
                        remaining=np.asarray(remaining, dtype=np.int32),
                        group=np.asarray(groups), domain=np.asarray(domains))
    print(json.dumps({"out": str(out), "episodes": n_ep, "windows": len(Xs),
                      "positive_fraction": float(y.mean()), "window": args.window,
                      "stride": args.stride}))


if __name__ == "__main__":
    main()
