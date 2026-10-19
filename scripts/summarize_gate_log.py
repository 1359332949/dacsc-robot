#!/usr/bin/env python3
"""把训练日志里的 GATE_METRICS JSON 提取成 CSV + 简要 Markdown 汇总。"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

PAT = re.compile(r"GATE_METRICS (\{.*\})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows, seen = [], set()
    for line in args.log.read_text(errors="ignore").splitlines():
        m = PAT.search(line)
        if not m:
            continue
        try:
            row = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        key = (row.get("step"), round(float(row.get("dp_loss", 0)), 8),
               row.get("layer"), row.get("mask_mode"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    if not rows:
        raise SystemExit("没有找到 GATE_METRICS")
    rows.sort(key=lambda r: (int(r.get("step", 0)), str(r.get("layer"))))
    out = args.out or args.log.with_suffix(".gate.csv")
    fields = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    first, last = rows[0], rows[-1]
    keys = ["step", "layer", "suppression", "v_ready", "v_raw_rel_std",
            "drop_iou_same", "drop_iou_diff", "drop_iou_gap",
            "score_cos_gap", "dp_loss", "gate_on_mse", "gate_off_mse",
            "gate_rel_delta"]
    print(f"rows={len(rows)} -> {out}")
    for r in (first, last):
        print("  " + " ".join(f"{k}={r.get(k)}" for k in keys if k in r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
