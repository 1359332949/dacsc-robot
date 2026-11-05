"""Validate complete per-task SR records before producing split summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacsc.eval_protocol import require_digest, validate_protocol
from dacsc.run_output import entry_file


def native_support(records: list[dict], key: str) -> dict:
    groups = {}
    for record in records:
        group = groups.setdefault(record[key], {"episodes": 0, "successes": 0})
        group["episodes"] += 1
        group["successes"] += int(record["success"])
    return {
        label: {**group, "success_rate": group["successes"] / group["episodes"]}
        for label, group in sorted(groups.items())
    }


def agg(dir_path: Path, expected_tasks: list[str], episodes: int) -> dict:
    if (
        episodes <= 0
        or not expected_tasks
        or len(set(expected_tasks)) != len(expected_tasks)
    ):
        raise ValueError("expected tasks must be unique and episodes must be positive")
    files = {p.stem: p for p in dir_path.glob("*.json")}
    if set(files) != set(expected_tasks):
        raise ValueError(
            f"{dir_path}: missing tasks {set(expected_tasks) - set(files)}, "
            f"unexpected tasks {set(files) - set(expected_tasks)}"
        )
    tasks = {}
    all_records = []
    identity = None
    for task in expected_tasks:
        payload = json.loads(files[task].read_text())
        protocol = payload["protocol"]
        validate_protocol(protocol)
        if identity is None:
            identity = protocol
        elif protocol != identity:
            raise ValueError(f"{task}: mixed checkpoint/condition/inference protocol")
        summary, records = payload["summary"], payload["records"]
        if (
            summary["task"] != task
            or summary["episodes"] != episodes
            or len(records) != episodes
        ):
            raise ValueError(f"{task}: task name or episode count mismatch")
        if {r["episode"] for r in records} != set(range(episodes)):
            raise ValueError(f"{task}: duplicate or missing episode IDs")
        if any(type(r["success"]) is not bool for r in records):
            raise ValueError(f"{task}: success must be boolean")
        if (
            summary["shift"] != protocol["shift"]
            or summary["blur_sigma"] != protocol["blur_sigma"]
            or summary["seed"] != protocol["base_seed"]
            or summary["layout_id"] != protocol["layout_id"]
            or len(summary["style_order"]) != episodes
        ):
            raise ValueError(f"{task}: summary condition differs from the protocol")
        if type(summary["max_steps"]) is not int or summary["max_steps"] <= 0:
            raise ValueError(f"{task}: missing/invalid task horizon")
        if not summary["styles"] or len(set(summary["styles"])) != len(
            summary["styles"]
        ):
            raise ValueError(f"{task}: invalid native style allowlist")
        for record in records:
            require_digest(record["manifest_sha256"])
            if (
                record["shift"] != protocol["shift"]
                or record["blur_sigma"] != protocol["blur_sigma"]
                or record["environment_seed"]
                != protocol["base_seed"] + record["episode"]
                or record["policy_seed"] != protocol["base_seed"] + record["episode"]
                or record["actual_layout"] != protocol["layout_id"]
                or record["requested_layout"] != protocol["layout_id"]
                or record["requested_style"]
                != summary["style_order"][record["episode"]]
                or record["actual_style"]
                != summary["style_order"][record["episode"]]
                or record["style_ids"] != summary["styles"]
                or record["actual_style"] not in summary["styles"]
                or not 0 <= record["steps"] <= summary["max_steps"]
            ):
                raise ValueError(f"{task}: episode does not match its frozen protocol")
        successes = sum(r["success"] for r in records)
        if summary["successes"] != successes or not math.isclose(
            summary["success_rate"], successes / episodes
        ):
            raise ValueError(f"{task}: summary does not match episode records")
        tasks[task] = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "max_steps": summary["max_steps"],
            "styles": summary["styles"],
            "per_style": native_support(records, "actual_style"),
            "per_layout": native_support(records, "actual_layout"),
            "manifest_sha256": [
                r["manifest_sha256"]
                for r in sorted(records, key=lambda r: r["episode"])
            ],
        }
        all_records.extend({"task": task, **record} for record in records)
    n = episodes * len(tasks)
    s = sum(t["successes"] for t in tasks.values())
    return {
        "episodes": n,
        "successes": s,
        "success_rate": s / n,
        "macro_task_success_rate": float(
            np.mean([t["success_rate"] for t in tasks.values()])
        ),
        "tasks": tasks,
        "per_style": native_support(all_records, "actual_style"),
        "per_layout": native_support(all_records, "actual_layout"),
        "protocol": identity,
        "records": all_records,
    }


def aggregate(root: Path, splits: list[str], tasks: list[str], episodes: int) -> dict:
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("splits must be nonempty and unique")
    out = {split: agg(root / split, tasks, episodes) for split in splits}
    identity = out[splits[0]]["protocol"]
    style_ranges = {
        "source": list(range(11, 31)),
        "val": list(range(31, 41)),
        "test": list(range(41, 51)),
    }
    for split in splits:
        if split not in style_ranges or out[split]["protocol"] != identity:
            raise ValueError(f"{split}: invalid split or mixed experiment identity")
        for task in tasks:
            current = out[split]["tasks"][task]
            if current["styles"] != style_ranges[split]:
                raise ValueError(
                    f"{task}/{split}: style allowlist violates the frozen split"
                )
            if current["max_steps"] != out[splits[0]]["tasks"][task]["max_steps"]:
                raise ValueError(f"{task}: task horizon differs across splits")
    for target in ("val", "test"):
        if "source" in out and target in out:
            out[f"gap_source_minus_{target}"] = (
                out["source"]["success_rate"] - out[target]["success_rate"]
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["source", "val", "test"],
        default=["source", "val"],
    )
    args = parser.parse_args()
    path = args.root / "summary.json"
    with entry_file(path):
        out = aggregate(args.root, args.splits, args.tasks, args.episodes)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(out, indent=2))
        temporary.replace(path)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
