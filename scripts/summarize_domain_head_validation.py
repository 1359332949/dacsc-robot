#!/usr/bin/env python3
"""Summarize DOMAIN_HEAD validation logs."""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path

PAT = re.compile(r"DOMAIN_HEAD (\{.*\})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--from-step", type=int, default=10)
    args = ap.parse_args()
    rows = []
    for log in sorted(args.root.glob("*/train.log")):
        vals = []
        for line in log.read_text(errors="ignore").splitlines():
            m = PAT.search(line)
            if not m:
                continue
            d = json.loads(m.group(1))
            if int(d.get("step", -1)) >= args.from_step:
                vals.append(d)
        if not vals:
            continue
        def mean(key):
            return sum(float(v[key]) for v in vals) / len(vals)
        def std(key):
            m = mean(key)
            return math.sqrt(sum((float(v[key]) - m) ** 2 for v in vals) / len(vals))
        row = {"run": log.parent.name, "n": len(vals)}
        for key in ("layout10_acc", "layout10_loss", "coarse3_acc", "coarse3_loss"):
            row[f"{key}_mean"] = mean(key)
            row[f"{key}_std"] = std(key)
        rows.append(row)
    if not rows:
        raise SystemExit("no DOMAIN_HEAD logs")
    for r in rows:
        print(r["run"], r["n"],
              "layout10", f"{r['layout10_acc_mean']:.3f}±{r['layout10_acc_std']:.3f}",
              "loss", f"{r['layout10_loss_mean']:.3f}",
              "coarse3", f"{r['coarse3_acc_mean']:.3f}±{r['coarse3_acc_std']:.3f}",
              "loss", f"{r['coarse3_loss_mean']:.3f}")
    print("acceptance: layout acc >0.30 (target >0.40), loss <2.30;"
          " coarse3 acc >0.50 (target >0.55), loss <1.10")


if __name__ == "__main__":
    main()
