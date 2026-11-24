"""Validate a filled failure-event CSV and emit annotations.json."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path

MODES = {"grasp_miss", "late_close", "slip", "no_lift", "wrong_target", "stuck", "other"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    errors = []
    with args.csv.open(newline="") as f:
        for row in csv.DictReader(f):
            ep = row["episode"].strip()
            if not ep: continue
            try:
                step = int(row["failure_event_step"])
            except (TypeError, ValueError):
                errors.append(f"{ep}: missing/non-integer failure_event_step")
                continue
            steps = int(row["steps"]) if row.get("steps") else None
            if step < 1 or (steps is not None and step > steps):
                errors.append(f"{ep}: failure_event_step out of range")
                continue
            mode = row.get("failure_mode", "").strip()
            if mode not in MODES:
                errors.append(f"{ep}: unknown failure_mode {mode!r}")
                continue
            out[ep] = {"failure_event_step": step, "failure_mode": mode,
                       "task_phase": row.get("task_phase", "").strip() or None,
                       "irrecoverable": (row.get("irrecoverable", "").strip().lower() in {"1", "true", "yes"}),
                       "event_confidence": row.get("event_confidence", "").strip() or None,
                       "event_source": "human_annotation"}
    if errors:
        raise SystemExit("annotation errors:\n" + "\n".join(errors))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "failure_event_annotations_v1", "episodes": out}, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "n_annotated": len(out)}))


if __name__ == "__main__":
    main()
