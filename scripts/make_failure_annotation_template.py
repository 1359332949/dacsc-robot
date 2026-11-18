"""Create a CSV template for failure-event annotation from trajectories.

The annotator fills failure_event_step and failure_mode. The script does not
invent labels.
"""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-dir", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for root in args.trajectory_dir:
        for p in sorted(root.rglob("s*_d*_seed*.json")):
            d = json.loads(p.read_text())
            if d.get("success"):
                continue
            rows.append({
                "episode": p.stem, "state_id": d["state_id"], "domain": d["domain"],
                "seed": d["seed"], "steps": d.get("steps"),
                "failure_event_step": "", "failure_mode": "",
                "task_phase": "", "irrecoverable": "", "event_confidence": "",
                "notes": "",
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["episode", "state_id", "domain", "seed", "steps",
                            "failure_event_step", "failure_mode", "task_phase",
                            "irrecoverable", "event_confidence", "notes"])
        w.writeheader(); w.writerows(rows)
    print(json.dumps({"out": str(args.out), "n_failed_episodes": len(rows)}))


if __name__ == "__main__":
    main()
