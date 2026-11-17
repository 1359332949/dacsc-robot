"""Audit whether existing trajectories can support a runtime failure monitor.

Read-only: no rollout, no model, no training.
"""

from __future__ import annotations
import argparse, json, statistics
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action-diag", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--source-pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    diag = json.loads(args.action_diag.read_text())
    split = json.loads(args.split_json.read_text())
    pairs = json.loads(args.source_pairs.read_text())
    recs = diag["records"]
    required = {
        "per_step_rgb": False, "per_step_visual_feature": False,
        "per_step_proprio": True, "per_chunk_index": True,
        "policy_output_action": True, "executed_action": True,
        "raw_preclip_action": False, "replan_timestamp": False,
        "final_success": True, "failure_event_time": False,
        "task_phase_label": "proxy_only", "object_ground_truth": False,
        "contact_state": False,
    }
    episodes = []
    for r in recs:
        actions = r.get("actions", []); states = r.get("states8", [])
        episodes.append({
            "state_id": r["state_id"], "domain": r["domain"], "seed": r["seed"],
            "success": r["success"], "steps": r["steps"],
            "n_actions": len(actions), "n_state8": len(states),
            "n_chunks_observed": len(states),
            "states_per_action": len(states) / max(len(actions), 1),
            "failure_phase": r.get("failure_phase"),
        })
    success = Counter((e["domain"], e["success"]) for e in episodes)
    phases = Counter((e["domain"], e["failure_phase"]) for e in episodes)
    lengths = [e["steps"] for e in episodes]
    state_counts = Counter(e["n_state8"] for e in episodes)
    summary = {
        "schema": "trajectory_observability_audit_v1",
        "n_episodes": len(episodes),
        "n_states": len(set(e["state_id"] for e in episodes)),
        "domains": sorted(set(e["domain"] for e in episodes)),
        "success": {f"{k[0]}:{k[1]}": v for k, v in success.items()},
        "steps_min": min(lengths), "steps_max": max(lengths),
        "steps_median": statistics.median(lengths),
        "n_state8_counts": dict(state_counts),
        "phase_counts": {f"{k[0]}:{k[1]}": v for k, v in phases.items()},
        "required_fields": required,
        "available": {
            "per_step_action": all(e["n_actions"] == e["steps"] for e in episodes),
            "per_chunk_state8": all(e["states_per_action"] < 1 for e in episodes),
            "per_step_state8": all(e["n_state8"] == e["steps"] for e in episodes),
        },
        "split": {
            "source_fit": len(split.get("source_train_state_ids", [])),
            "source_validation": len(split.get("source_val_state_ids", [])),
            "target_dev": len(split.get("target_dev_state_ids", [])),
            "target_test": len(split.get("target_test_state_ids", [])),
            "target_test_status": "already used for exploratory comparisons; not confirmatory",
        },
        "source_pairs": {
            "n_states": len(pairs.get("records", [])),
            "scope": "first chunk only; used source_fit + source_validation; known-perturbation calibration",
        },
        "monitor_training_ready": False,
        "blockers": [
            "no per-step RGB or frozen visual features",
            "no per-step executed-vs-policy action record",
            "no raw pre-clip action",
            "no failure-event timestamp; only final success",
            "task phase is a heuristic proxy, not a verified label",
            "no object/contact ground truth",
            "no recovery-interface validation",
            "target test already exploratory; no fresh confirmatory set",
        ],
        "warning_lead_time_computable": False,
        "recovery_interface_tested": False,
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
