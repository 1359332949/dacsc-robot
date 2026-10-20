#!/usr/bin/env python3
"""Summarize flow K48 no-mask/random/domain matrix."""
from __future__ import annotations
import argparse, json, math, re
from collections import defaultdict
from pathlib import Path

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
        if group == "flow_no_mask":
            jl = d / "logs.json.txt"
            last = None
            if jl.exists():
                for line in jl.read_text(errors="ignore").splitlines():
                    try:
                        x = json.loads(line)
                    except Exception:
                        continue
                    if "train_action_mse_error" in x:
                        last = x
            if last is None:
                missing.append(d.name)
                continue
            rows.append({"group": group, "seed": seed,
                         "train_loss": last.get("train_loss"),
                         "action_mse": last.get("train_action_mse_error")})
        else:
            if gate is None:
                missing.append(d.name)
                continue
            rows.append({"group": group, "seed": seed,
                         **{k: gate.get(k) for k in KEYS}})
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    print(f"runs={len(rows)} missing={missing}")
    for group in sorted(by):
        print(f"\n[{group}] n={len(by[group])}")
        keys = ["drop_iou_gap", "drop_iou_sample_gap", "gate_rel_delta",
                "v_raw_rel_std", "eligible_frac", "train_loss", "action_mse"]
        for k in keys:
            vals = [float(r[k]) for r in by[group] if r.get(k) is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            print(f"  {k:24s} mean={mean:+.6f} std={std:.6f} values={vals}")


if __name__ == "__main__":
    main()
