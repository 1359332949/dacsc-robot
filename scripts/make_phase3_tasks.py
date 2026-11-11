"""Freeze Phase-3 task candidates and seeds (pilot selection is separate)."""

from __future__ import annotations

import glob
import json
from pathlib import Path

TASKS = Path("/hy-tmp/validation/dim_sweep_tasks.json")
OUT = Path("configs/phase3_tasks.json")
DOMAINS = ["Camera", "Background", "Noise", "Light"]
SUITES = ["libero_spatial", "libero_object"]


def baseline_ep0() -> dict[tuple[str, int], bool]:
    out = {}
    for f in glob.glob("/hy-tmp/validation/lpfull_*/libero_*.jsonl"):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["suite"], int(r["task_id"]))
            if r["episode_id"] == 0 and key not in out:
                out[key] = bool(r["success"])
    return out


def main() -> None:
    blob = json.loads(TASKS.read_text())
    base = baseline_ep0()
    cells = {}
    for dim in DOMAINS:
        for suite in SUITES:
            items = [it for it in blob[dim] if it["suite"] == suite]
            successes = [it["task_id"] for it in items if base.get((suite, it["task_id"]))]
            failures = [it["task_id"] for it in items if not base.get((suite, it["task_id"]))]
            # mix easy and hard candidates so the pilot can find tasks in the 20-80% band
            cands = sorted(successes)[:3] + sorted(failures)[:3]
            cells[f"{dim}|{suite}"] = {"domain": dim, "suite": suite, "candidates": cands}
    cfg = {
        "phase": 3,
        "frozen_config": {
            "checkpoint": "repos/CoRP/checkpoints/corp_libero.pt",
            "a_basis": "/hy-tmp/validation/interface_stats_spatial_k32_big2.json",
            "eta": 0.15, "adapt_steps": 5, "draws": 8, "lr": 0.06,
            "exec_len": 6, "num_steps": 10, "horizon": 12,
            "lambda_protect": 5.0,
            "lambda_calibration": "mean(||grad_salt||/||grad_pen||)=5.88 on 8 real clean batches, snapped to grid",
            "trigger": "always-on",
            "pilot_seeds": [202, 203, 204, 205],
            "formal_seeds": [7, 8, 9, 10, 11],
            "episodes_per_task": 5,
        },
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cfg, indent=2))
    print(json.dumps({"cells": {k: v["candidates"] for k, v in cells.items()}}, indent=2))


if __name__ == "__main__":
    main()
