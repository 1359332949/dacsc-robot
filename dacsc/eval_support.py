"""Task-level train support checks for closed-loop source evaluation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from dacsc.corp_data import read_episode_meta


def task_release(data_root: Path, task: str) -> Path:
    task_root = data_root / "v1.0/pretrain/atomic" / task
    releases = sorted(task_root.glob("*/lerobot"))
    if len(releases) != 1:
        raise ValueError(
            f"{task}: expected exactly one RoboCasa pretrain release, "
            f"found {len(releases)}"
        )
    return releases[0]


def source_task_support(
    dataset_path: Path,
    *,
    styles: list[int],
    layout_id: int,
    train_style_range: tuple[int, int],
    source_holdout_residue: int,
    episode_mod: int = 10,
) -> dict:
    """Count train-split styles and the requested fixed-layout panel."""
    if (
        not styles
        or len(styles) != len(set(styles))
        or any(type(style) is not int or not 1 <= style <= 60 for style in styles)
    ):
        raise ValueError("styles must be a non-empty unique integer list within 1..60")
    if type(layout_id) is not int or not 11 <= layout_id <= 60:
        raise ValueError("pretrain layout_id must be an integer within 11..60")
    lo, hi = train_style_range
    if not 11 <= lo <= hi <= 60:
        raise ValueError("train_style_range must be within 11..60")
    if not 0 <= source_holdout_residue < episode_mod:
        raise ValueError("source holdout residue must be within the episode modulus")

    metadata = read_episode_meta(dataset_path)
    train = {
        episode: item
        for episode, item in metadata.items()
        if lo <= item["style_id"] <= hi
        and episode % episode_mod != source_holdout_residue
    }
    style_counts = Counter(item["style_id"] for item in train.values())
    layout_counts = Counter(item["layout_id"] for item in train.values())
    pair_counts = Counter(
        (item["style_id"], item["layout_id"]) for item in train.values()
    )
    requested_pairs = [(style, layout_id) for style in styles]
    missing_styles = [style for style in styles if style_counts[style] == 0]
    return {
        "dataset": str(dataset_path),
        "episodes_total": len(metadata),
        "train_episodes": len(train),
        "train_style_range": [lo, hi],
        "source_holdout_residue": source_holdout_residue,
        "requested_styles": styles,
        "requested_layout": layout_id,
        "style_train_support": {
            str(style): style_counts[style] for style in styles
        },
        "layout_train_support": layout_counts[layout_id],
        "requested_pair_support": {
            f"style={style},layout={layout_id}": pair_counts[(style, layout_id)]
            for style in styles
        },
        "missing_train_styles": missing_styles,
        "layout_supported": layout_counts[layout_id] > 0,
        "styles_supported": not missing_styles,
        "pairs_supported": all(pair_counts[pair] > 0 for pair in requested_pairs),
    }


def validate_source_panel(
    data_root: Path,
    *,
    tasks: list[str],
    styles: list[int],
    layout_id: int,
    train_style_range: tuple[int, int],
    source_holdout_residue: int,
    require_pairs: bool = False,
) -> dict:
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("tasks must be a non-empty unique list")
    tasks = sorted(tasks)
    reports = {}
    for task in tasks:
        report = source_task_support(
            task_release(data_root, task),
            styles=styles,
            layout_id=layout_id,
            train_style_range=train_style_range,
            source_holdout_residue=source_holdout_residue,
        )
        report["passed"] = report["layout_supported"] and report["styles_supported"]
        if require_pairs:
            report["passed"] = report["passed"] and report["pairs_supported"]
        reports[task] = report
    return {
        "schema": "dacsc_source_panel_support_v1",
        "data_root": str(data_root),
        "tasks": reports,
        "passed": all(report["passed"] for report in reports.values()),
    }
