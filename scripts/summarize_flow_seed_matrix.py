#!/usr/bin/env python3
"""Summarize flow K48 no-mask/random/domain matrix."""
from __future__ import annotations
import argparse, json, math, re
from collections import defaultdict
from pathlib import Path

from metric_utils import read_run_metrics

PAT = re.compile(r"GATE_METRICS (\{.*\})")
KEYS = ["drop_iou_gap", "drop_iou_sample_gap", "gate_rel_delta",
        "v_raw_rel_std", "eligible_frac"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    rows, missing = [], []
    for d in sorted(args.root.iterdir()):
        if not d.is_dir():
            continue
        m = re.fullmatch(r"(.+)_s(\d+)", d.name)
        if not m:
            continue
        group, seed = m.group(1), int(m.group(2))
        log = d / "train.log"
        gate = None
        for line in log.read_text(errors="ignore").splitlines():
            mm = PAT.search(line)
            if not mm:
                continue
            try:
                x = json.loads(mm.group(1))
            except Exception:
                continue
            if int(x.get("step", -1)) == 50:
                gate = x
        metrics = read_run_metrics(d)
        if group != "flow_no_mask" and gate is None:
            missing.append(d.name)
            continue
        if metrics["val_action_mse"] is None:
            print(f"[missing val_action_mse] {d.name}")
        if metrics["test_action_mse"] is None:
            print(f"[missing test_action_mse] {d.name}")
        row = {"group": group, "seed": seed, **metrics}
        if gate is not None:
            row.update({k: gate.get(k) for k in KEYS})
        rows.append(row)
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    print(f"runs={len(rows)} missing={missing}")
    for group in sorted(by):
        print(f"\n[{group}] n={len(by[group])}")
        keys = ["drop_iou_gap", "drop_iou_sample_gap", "gate_rel_delta",
                "v_raw_rel_std", "eligible_frac", "val_action_mse",
                "test_action_mse", "train_action_mse", "train_loss"]
        for k in keys:
            vals = [float(r[k]) for r in by[group] if r.get(k) is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            print(f"  {k:24s} mean={mean:+.6f} std={std:.6f} values={vals}")


if __name__ == "__main__":
    main()
