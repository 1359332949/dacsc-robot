"""Select 3 tasks per cell from pilot results (frozen success rate in 20-80%)."""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

PILOT = Path("/hy-tmp/validation/p3_pilot2")
OUT = Path("configs/phase3_final_tasks.json")
CELLS = {
    "Camera|libero_spatial": ("libero_spatial", "Camera"),
    "Camera|libero_object": ("libero_object", "Camera"),
    "Background|libero_spatial": ("libero_spatial", "Background"),
    "Background|libero_object": ("libero_object", "Background"),
    "Noise|libero_spatial": ("libero_spatial", "Noise"),
    "Noise|libero_object": ("libero_object", "Noise"),
    "Light|libero_spatial": ("libero_spatial", "Light"),
    "Light|libero_object": ("libero_object", "Light"),
}


def main() -> None:
    pilot_cfg = json.loads(Path("configs/phase3_pilot_tasks.json").read_text())
    cells = {}
    for cell, (suite, domain) in CELLS.items():
        rows = [json.loads(l) for l in open(PILOT / f"{cell.replace('|','_')}.jsonl") if l.strip()]
        per_task = defaultdict(list)
        for r in rows:
            per_task[r["task_id"]].append(bool(r["success"]))
        rates = {t: (sum(v), len(v)) for t, v in per_task.items()}
        in_band = [t for t, (s, n) in rates.items() if n >= 2 and 0.2 <= s / n <= 0.8]
        # rank by closeness to 0.5 among in-band tasks, then by id
        in_band.sort(key=lambda t: abs(rates[t][0] / rates[t][1] - 0.5))
        chosen = in_band[:3]
        if len(chosen) < 3:
            rest = [t for t in rates if t not in chosen]
            rest.sort(key=lambda t: abs(rates[t][0] / max(rates[t][1], 1) - 0.5))
            chosen += rest[: 3 - len(chosen)]
        cells[cell] = {
            "domain": domain, "suite": suite, "tasks": chosen,
            "pilot_rates": {str(t): f"{rates[t][0]}/{rates[t][1]}" for t in chosen},
            "in_band": len(in_band),
        }
        print(cell, "chosen", chosen, {str(t): f"{rates[t][0]}/{rates[t][1]}" for t in chosen},
              f"in_band={len(in_band)}")
    cfg = {
        "phase": 3,
        "frozen_config": {
            "checkpoint": "repos/CoRP/checkpoints/corp_libero.pt",
            "a_basis": "/hy-tmp/validation/interface_stats_spatial_k32_big2.json",
            "eta": 0.15, "adapt_steps": 5, "draws": 8, "lr": 0.06,
            "exec_len": 6, "num_steps": 10, "horizon": 12,
            "lambda_protect": 5.0,
            "lambda_calibration": "mean(||grad_salt||/||grad_pen||)=5.88 on 8 real clean batches (grid-snapped)",
            "trigger": "always-on",
            "pilot_seeds": [202, 203, 204, 205],
            "formal_seeds": [7, 8, 9, 10, 11],
            "episodes_per_task": 5,
            "methods": ["frozen", "salt", "sg_salt"],
        },
        "cells": cells,
    }
    OUT.write_text(json.dumps(cfg, indent=2))
    print("written", OUT)


if __name__ == "__main__":
    main()
