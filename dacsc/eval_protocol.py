"""Experiment identity for SR aggregation (DomainBed-style grouping by arguments)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from dacsc.corp_images import image_contract
from dacsc.upstream import verify_upstream

PROTOCOL_VERSION = "corp_robocasa_sr_v10"
TARGET_PAIRS = tuple((index, index) for index in range(1, 11))
DEFAULT_STYLE_RANGES = {
    "source": list(range(11, 51)),
    "val": list(range(51, 61)),
    "target": list(range(1, 11)),
}
PROTOCOL_KEYS = {
    "version",
    "checkpoint_sha256",
    "scene_split",
    "shift",
    "blur_sigma",
    "layouts",
    "layout_id",
    "target_pairs",
    "base_seed",
    "action_steps",
    "action_horizon",
    "euler_steps",
    "image_size",
    "precision",
    "horizon_source",
    "images",
    "upstream",
    "training",
}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_digest(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("expected a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError("expected a SHA256 digest") from exc


def expected_evaluation_styles(origin: dict, split: str) -> list[int]:
    """Return the only style range allowed by a checkpoint training origin."""
    validate_training_origin(origin)
    contract = origin["data_contract"]
    if split == "source":
        return list(range(contract["train_styles"][0], contract["train_styles"][1] + 1))
    if split == "val":
        return list(range(contract["val_styles"][0], contract["val_styles"][1] + 1))
    if split == "target":
        target_styles = list(range(1, 11))
        train = set(range(contract["train_styles"][0], contract["train_styles"][1] + 1))
        val = set(range(contract["val_styles"][0], contract["val_styles"][1] + 1))
        if set(target_styles) & (train | val):
            raise ValueError("target styles overlap checkpoint train/val styles")
        return target_styles
    raise ValueError(f"unknown evaluation split: {split}")


def validate_protocol(protocol: dict) -> None:
    if set(protocol) != PROTOCOL_KEYS or protocol["version"] != PROTOCOL_VERSION:
        raise ValueError(
            "SR result has an unsupported/incomplete protocol; rerun evaluation"
        )
    require_digest(protocol["checkpoint_sha256"])
    validate_training_origin(protocol["training"])
    if protocol["images"] != image_contract():
        raise ValueError(
            "image implementation/parameters differ from the frozen protocol"
        )
    if protocol["upstream"] != verify_upstream():
        raise ValueError("upstream sources differ from the frozen protocol")
    if (
        protocol["horizon_source"]
        != "robocasa.utils.dataset_registry_utils.get_task_horizon"
    ):
        raise ValueError("SR budget must come from the official RoboCasa task registry")
    if protocol["shift"] not in {
        "none",
        "grayscale",
        "color_jitter",
        "low_contrast",
        "hue_shift",
    }:
        raise ValueError("unknown appearance condition")
    if (
        not math.isfinite(protocol["blur_sigma"])
        or protocol["blur_sigma"] < 0
        or not protocol["layouts"]
        or type(protocol["base_seed"]) is not int
        or protocol["precision"] not in {"fp32", "bf16"}
    ):
        raise ValueError("invalid blur/layout protocol")
    if protocol["layouts"] != sorted(set(protocol["layouts"])):
        raise ValueError("layouts must be sorted and unique")
    scene_split = protocol["scene_split"]
    if scene_split == "pretrain":
        if (
            type(protocol["layout_id"]) is not int
            or any(layout < 11 or layout > 60 for layout in protocol["layouts"])
            or protocol["layout_id"] not in protocol["layouts"]
            or protocol["target_pairs"] is not None
        ):
            raise ValueError("invalid pretrain scene/layout protocol")
    elif scene_split == "target":
        expected_pairs = [list(pair) for pair in TARGET_PAIRS]
        if (
            protocol["layout_id"] is not None
            or protocol["layouts"] != list(range(1, 11))
            or protocol["target_pairs"] != expected_pairs
        ):
            raise ValueError("invalid official target scene protocol")
    else:
        raise ValueError("unknown scene split")
    if any(
        type(protocol[key]) is not int or protocol[key] <= 0
        for key in ("action_steps", "action_horizon", "euler_steps", "image_size")
    ):
        raise ValueError("invalid inference budget")
    if protocol["action_steps"] > protocol["action_horizon"]:
        raise ValueError("action chunk execution exceeds the prediction horizon")


def validate_training_origin(origin: dict) -> None:
    expected = {
        "seed",
        "budget",
        "optimization",
        "initialization_sha256",
        "data_metadata_sha256",
        "data_length",
        "intervention",
        "pixel_augmentation",
        "task_aux",
        "initialization_policy",
        "data_contract",
        "completed_steps",
    }
    if (
        set(origin) != expected
        or type(origin["seed"]) is not int
        or type(origin["data_length"]) is not int
        or origin["data_length"] <= 0
        or type(origin["completed_steps"]) is not int
        or origin["completed_steps"] <= 0
        or not isinstance(origin["budget"], dict)
        or type(origin["budget"].get("steps")) is not int
        or origin["completed_steps"] > origin["budget"]["steps"]
    ):
        raise ValueError("missing/invalid training origin for seed-level comparison")
    require_digest(origin["initialization_sha256"])
    require_digest(origin["data_metadata_sha256"])
    if (
        not isinstance(origin["budget"], dict)
        or not origin["budget"]
        or not isinstance(origin["optimization"], dict)
        or not origin["optimization"]
    ):
        raise ValueError("training budget and optimization recipe are required")
    intervention = origin["intervention"]
    if intervention is not None and (
        not isinstance(intervention, dict)
        or intervention.get("donor_seed") != origin["seed"]
    ):
        raise ValueError(
            "intervention donor seed must match its recorded training seed"
        )
    pixel = origin["pixel_augmentation"]
    if pixel is not None:
        from dacsc.pixel_augmentation import pixel_augmentation_contract

        if (
            pixel != pixel_augmentation_contract("color_jitter", origin["seed"])
            or intervention is not None
        ):
            raise ValueError(
                "invalid training pixel recipe or unsupported combined interventions"
            )
    task_aux = origin["task_aux"]
    if task_aux is not None:
        expected_task_aux = {
            "num_classes",
            "loss_weight",
            "depths",
            "head_hidden_dim",
            "task_to_id",
            "implementation",
        }
        if (
            not isinstance(task_aux, dict)
            or set(task_aux) != expected_task_aux
            or type(task_aux["num_classes"]) is not int
            or task_aux["num_classes"] < 2
            or not isinstance(task_aux["loss_weight"], (int, float))
            or not math.isfinite(task_aux["loss_weight"])
            or task_aux["loss_weight"] <= 0
            or not isinstance(task_aux["depths"], (list, tuple))
            or not task_aux["depths"]
            or any(type(depth) is not int or depth < 0 for depth in task_aux["depths"])
            or type(task_aux["head_hidden_dim"]) is not int
            or task_aux["head_hidden_dim"] <= 0
            or not isinstance(task_aux["task_to_id"], dict)
            or len(task_aux["task_to_id"]) != task_aux["num_classes"]
            or not isinstance(task_aux["implementation"], str)
            or not task_aux["implementation"]
        ):
            raise ValueError("invalid task auxiliary training origin")
    if (
        not isinstance(origin["initialization_policy"], str)
        or not origin["initialization_policy"]
    ):
        raise ValueError("missing initialization policy in training origin")
    contract = origin["data_contract"]
    expected_contract = {
        "tasks",
        "train_styles",
        "val_styles",
        "source_holdout_residue",
        "train",
        "source",
        "val",
        "project_git_commit",
        "project_git_dirty",
    }
    if not isinstance(contract, dict) or set(contract) != expected_contract:
        raise ValueError("missing/invalid training data contract")
    tasks = contract["tasks"]
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(task, str) or not task for task in tasks)
        or tasks != sorted(set(tasks))
    ):
        raise ValueError("training task list must be sorted and unique")
    for key in ("train_styles", "val_styles"):
        value = contract[key]
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(item) is not int or item < 1 or item > 60 for item in value)
            or value[0] > value[1]
        ):
            raise ValueError(f"invalid {key} range")
    if set(range(contract["train_styles"][0], contract["train_styles"][1] + 1)) & set(
        range(contract["val_styles"][0], contract["val_styles"][1] + 1)
    ):
        raise ValueError("training and validation style ranges must be disjoint")
    if (
        type(contract["source_holdout_residue"]) is not int
        or not 0 <= contract["source_holdout_residue"] < 10
    ):
        raise ValueError("invalid source holdout residue")
    for split in ("train", "source", "val"):
        support = contract[split]
        if not isinstance(support, dict) or set(support) != {
            "frames",
            "episodes",
            "metadata_sha256",
            "selection_sha256",
        }:
            raise ValueError(f"invalid {split} data support")
        if (
            type(support["frames"]) is not int
            or support["frames"] <= 0
            or type(support["episodes"]) is not int
            or support["episodes"] <= 0
        ):
            raise ValueError(f"invalid {split} frame/episode support")
        require_digest(support["metadata_sha256"])
        require_digest(support["selection_sha256"])
    commit = contract["project_git_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise ValueError("invalid project git commit")
    if type(contract["project_git_dirty"]) is not bool:
        raise ValueError("invalid project git dirty state")
