"""Load the frozen V10 pilot contract as the source of launcher defaults."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/v10_atomic_target_pilot.json"


def _style_range(value, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int for item in value)
        or not 1 <= value[0] <= value[1] <= 60
    ):
        raise ValueError(f"{label} must be an inclusive [lo, hi] style range")
    return list(range(value[0], value[1] + 1))


def load_v10_config(path: Path = CONFIG_PATH) -> dict:
    """Read and validate the machine-readable V10 contract."""
    config = json.loads(Path(path).read_text())
    if config.get("protocol") != "corp_robocasa_v10_atomic_target_pilot":
        raise ValueError("unsupported V10 protocol identity")
    tasks = config.get("train_tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or tasks != sorted(set(tasks))
        or any(not isinstance(task, str) or not task for task in tasks)
    ):
        raise ValueError("V10 train_tasks must be sorted and unique")

    pretrain = config.get("pretrain")
    target = config.get("target")
    budget = config.get("pilot_budget")
    evaluation = config.get("evaluation")
    if not all(isinstance(value, dict) for value in (pretrain, target, budget, evaluation)):
        raise ValueError("V10 config is missing a required section")

    train = _style_range(pretrain.get("train_styles"), "pretrain.train_styles")
    source = _style_range(
        pretrain.get("source_pilot_styles"), "pretrain.source_pilot_styles"
    )
    val = _style_range(pretrain.get("val_styles"), "pretrain.val_styles")
    target_styles = _style_range(target.get("styles"), "target.styles")
    if not set(source).issubset(train):
        raise ValueError("source pilot styles must be a subset of training styles")
    if set(train) & set(val) or set(train) & set(target_styles):
        raise ValueError("train, internal-val and target styles must be disjoint")
    if target_styles != list(range(1, 11)):
        raise ValueError("V10 target styles must remain 1-10")
    if target.get("scene_split") != "target" or target.get("pairs") != [
        [index, index] for index in range(1, 11)
    ]:
        raise ValueError("V10 target must use the official paired target scenes")

    positive = (
        "steps",
        "batch_size",
        "task_aux_depth",
        "seed",
        "per_task_eval",
        "episodes_per_task",
        "layout_id",
        "action_steps",
    )
    for key in positive:
        section = budget if key in budget else evaluation
        if type(section.get(key)) is not int or section[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("lr", "ema_decay", "task_aux_lambda"):
        if not isinstance(budget.get(key), (int, float)):
            raise ValueError(f"{key} must be numeric")
    if not 0 <= budget["ema_decay"] < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    cache_tasks = budget.get("cache_video_tasks")
    if (
        not isinstance(cache_tasks, list)
        or any(task not in tasks for task in cache_tasks)
        or len(cache_tasks) != len(set(cache_tasks))
    ):
        raise ValueError("cache_video_tasks must uniquely name configured tasks")
    return config


def checkpoint_style_ranges(config: dict | None = None) -> dict[str, list[int]]:
    """Styles represented by the training/offline-evaluation contract."""
    config = load_v10_config() if config is None else config
    return {
        "source": _style_range(
            config["pretrain"]["train_styles"], "pretrain.train_styles"
        ),
        "val": _style_range(config["pretrain"]["val_styles"], "pretrain.val_styles"),
        "target": _style_range(config["target"]["styles"], "target.styles"),
    }


def closed_loop_style_ranges(config: dict | None = None) -> dict[str, list[int]]:
    """Styles used by the frozen closed-loop V10 source/val/target panel."""
    config = load_v10_config() if config is None else config
    return {
        "source": _style_range(
            config["pretrain"]["source_pilot_styles"],
            "pretrain.source_pilot_styles",
        ),
        "val": _style_range(config["pretrain"]["val_styles"], "pretrain.val_styles"),
        "target": _style_range(config["target"]["styles"], "target.styles"),
    }


def _range_text(value: list[int]) -> str:
    if value != list(range(value[0], value[-1] + 1)):
        raise ValueError("shell style defaults must be contiguous")
    return f"{value[0]}-{value[-1]}"


def shell_defaults(config: dict | None = None) -> dict[str, str]:
    """Return quoted shell assignments used by the thin V10 launchers."""
    config = load_v10_config() if config is None else config
    budget = config["pilot_budget"]
    evaluation = config["evaluation"]
    styles = closed_loop_style_ranges(config)
    train = config["pretrain"]["train_styles"]
    val = config["pretrain"]["val_styles"]
    return {
        "V10_TASKS": " ".join(config["train_tasks"]),
        "V10_STEPS": str(budget["steps"]),
        "V10_BATCH_SIZE": str(budget["batch_size"]),
        "V10_LR": str(budget["lr"]),
        "V10_EMA_DECAY": str(budget["ema_decay"]),
        "V10_TASK_AUX_LAMBDA": str(budget["task_aux_lambda"]),
        "V10_TASK_AUX_DEPTH": str(budget["task_aux_depth"]),
        "V10_SEED": str(budget["seed"]),
        "V10_SAVE_EVERY": str(budget["save_every"]),
        "V10_PER_TASK_EVAL": str(budget["per_task_eval"]),
        "V10_CACHE_VIDEO_TASKS": " ".join(budget["cache_video_tasks"]),
        "V10_TRAIN_STYLE_MIN": str(train[0]),
        "V10_TRAIN_STYLE_MAX": str(train[1]),
        "V10_VAL_STYLE_MIN": str(val[0]),
        "V10_VAL_STYLE_MAX": str(val[1]),
        "V10_SOURCE_HOLDOUT_RESIDUE": str(
            config["pretrain"]["source_holdout_residue"]
        ),
        "V10_SOURCE_STYLES": _range_text(styles["source"]),
        "V10_VAL_STYLES": _range_text(styles["val"]),
        "V10_TARGET_STYLES": _range_text(styles["target"]),
        "V10_EPISODES": str(evaluation["episodes_per_task"]),
        "V10_LAYOUT_ID": str(evaluation["layout_id"]),
        "V10_ACTION_STEPS": str(evaluation["action_steps"]),
    }


def shell_text(config: dict | None = None) -> str:
    return "\n".join(
        f"{key}={shlex.quote(str(value))}"
        for key, value in shell_defaults(config).items()
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shell",
        action="store_true",
        help="print shell assignments instead of normalized JSON",
    )
    args = parser.parse_args(argv)
    config = load_v10_config()
    print(shell_text(config) if args.shell else json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
