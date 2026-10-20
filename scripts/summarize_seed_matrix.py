#!/usr/bin/env python3
"""汇总 random / D / F × seed 的 step50 机制指标。"""
from __future__ import annotations
import argparse, json, math, re
from collections import defaultdict
from pathlib import Path

PAT = re.compile(r"GATE_METRICS (\{.*\})")
KEYS = ["drop_iou_same", "drop_iou_diff", "drop_iou_gap",
        "drop_iou_sample_gap", "gate_rel_delta", "v_raw_rel_std", "dp_loss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    for log in sorted(args.root.glob("*/train.log")):
        run = log.parent.name
        m = re.fullmatch(r"(.+)_s(\d+)", run)
        if not m:
            continue
        group, seed = m.group(1), int(m.group(2))
        found = None
        for line in log.read_text(errors="ignore").splitlines():
            mm = PAT.search(line)
            if not mm:
                continue
            try:
                d = json.loads(mm.group(1))
            except Exception:
                continue
            if int(d.get("step", -1)) == args.step:
                found = d
        train_action_mse = None
        train_loss = None
        logs = log.parent / "logs.json.txt"
        if logs.exists():
            for line in logs.read_text(errors="ignore").splitlines():
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("train_action_mse_error") is not None:
                    train_action_mse = float(d["train_action_mse_error"])
                if d.get("train_loss") is not None:
                    train_loss = float(d["train_loss"])
        if found is None:
            print(f"[missing step{args.step}] {run}")
        row = {"group": group, "seed": seed,
               "train_action_mse": train_action_mse, "train_loss": train_loss}
        if found is not None:
            row.update({k: found.get(k) for k in KEYS})
        rows.append(row)

    if args.csv is not None:
        import csv
        fields = ["group", "seed", *KEYS, "train_action_mse", "train_loss"]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sorted(rows, key=lambda x: (x["group"], x["seed"])):
                w.writerow({k: r.get(k) for k in fields})
        print(f"wrote {args.csv}")

    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    keys = ["drop_iou_gap", "drop_iou_sample_gap", "gate_rel_delta",
            "gate_on_mse", "gate_off_mse", "v_raw_rel_std", "dp_loss",
            "train_action_mse", "train_loss"]
    print(f"runs={len(rows)} step={args.step}")
    for group in sorted(by):
        print(f"\n[{group}] n={len(by[group])}")
        for k in keys:
            vals = [float(r[k]) for r in by[group] if r.get(k) is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
            print(f"  {k:24s} mean={mean:+.6f} std={std:.6f} values={vals}")


if __name__ == "__main__":
    main()
