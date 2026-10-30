"""Standard MSE aggregation from PyTorch squared errors; no custom weighting."""

from __future__ import annotations

import numpy as np

MSE_SCHEMA = "corp_action_mse_v1"
ACTION_GROUPS = {
    "eef_position": slice(0, 3),
    "eef_rotation": slice(3, 6),
    "gripper_close": slice(6, 7),
    "base_motion": slice(7, 11),
    "control_mode": slice(11, 12),
}
ACTION_DIMENSIONS = [
    "eef_position_0",
    "eef_position_1",
    "eef_position_2",
    "eef_rotation_0",
    "eef_rotation_1",
    "eef_rotation_2",
    "gripper_close",
    "base_motion_0",
    "base_motion_1",
    "base_motion_2",
    "base_motion_3",
    "control_mode",
]


def action_mse_summary(records: list[dict]) -> dict:
    """Counts and sums retain the original masked scalar MSE definition.

    Per-episode rows are bookkeeping for clustered analysis; frames and future
    chunk positions are never labelled as independent episodes.
    """
    if not records:
        raise ValueError("no action error records")
    seen = set()
    episodes = {}
    tasks = {}
    total = np.zeros(12, dtype=np.float64)
    positions = 0
    for row in records:
        for key in ("episode", "frame", "style_id", "valid_action_positions"):
            if type(row[key]) is not int:
                raise ValueError(f"{key} must be an integer")
        if (
            not isinstance(row["task"], str)
            or not row["task"]
            or row["episode"] < 0
            or row["frame"] < 0
            or not 11 <= row["style_id"] <= 60
            or row["valid_action_positions"] <= 0
        ):
            raise ValueError("invalid action sample identity/support")
        key = (row["task"], row["episode"], row["frame"])
        if key in seen:
            raise ValueError("duplicate evaluation frame")
        seen.add(key)
        sse = np.asarray(row["squared_error_sum"], dtype=np.float64)
        if sse.shape != (12,) or not np.isfinite(sse).all() or (sse < 0).any():
            raise ValueError("expected twelve finite, nonnegative squared-error sums")
        n = row["valid_action_positions"]
        total += sse
        positions += n
        task = tasks.setdefault(
            row["task"],
            {"sse": np.zeros(12), "positions": 0, "samples": 0, "episodes": set()},
        )
        task["sse"] += sse
        task["positions"] += n
        task["samples"] += 1
        task["episodes"].add(row["episode"])
        episode = episodes.setdefault(
            (row["task"], row["episode"]),
            {
                "task": row["task"],
                "episode": row["episode"],
                "style_id": row["style_id"],
                "sse": np.zeros(12),
                "valid_action_positions": 0,
                "frames": [],
            },
        )
        if episode["style_id"] != row["style_id"]:
            raise ValueError("one episode cannot change native style")
        episode["sse"] += sse
        episode["valid_action_positions"] += n
        episode["frames"].append(row["frame"])
    per_task = {
        name: float(np.sum(t["sse"]) / (12 * t["positions"]))
        for name, t in sorted(tasks.items())
    }
    return {
        "schema": MSE_SCHEMA,
        "metric_implementation": "torch.nn.functional.mse_loss(reduction='none'); NumPy sum/mean",
        "action_space": "raw_unclipped_action_commands",
        "micro_action_mse": float(np.sum(total) / (12 * positions)),
        "macro_task_action_mse": float(np.mean(list(per_task.values()))),
        "per_action_group_mse": {
            name: float(np.sum(total[sl]) / ((sl.stop - sl.start) * positions))
            for name, sl in ACTION_GROUPS.items()
        },
        "per_action_dimension_mse": dict(
            zip(ACTION_DIMENSIONS, (total / positions).tolist(), strict=True)
        ),
        "per_task_action_mse": per_task,
        "per_task_support": {
            name: {
                "samples": t["samples"],
                "episodes": len(t["episodes"]),
                "valid_action_positions": t["positions"],
            }
            for name, t in sorted(tasks.items())
        },
        "samples": len(records),
        "episodes": len(episodes),
        "tasks": len(tasks),
        "valid_action_scalars": 12 * positions,
        "per_episode": [
            {
                "task": e["task"],
                "episode": e["episode"],
                "style_id": e["style_id"],
                "frames": sorted(e["frames"]),
                "valid_action_positions": e["valid_action_positions"],
                "squared_error_sum": e["sse"].tolist(),
                "micro_action_mse": float(
                    np.sum(e["sse"]) / (12 * e["valid_action_positions"])
                ),
            }
            for _, e in sorted(episodes.items())
        ],
        "records": sorted(records, key=lambda r: (r["task"], r["episode"], r["frame"])),
    }
