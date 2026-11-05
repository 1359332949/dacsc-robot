"""Experiment identity for SR aggregation (DomainBed-style grouping by arguments)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from dacsc.corp_images import image_contract
from dacsc.upstream import verify_upstream

PROTOCOL_VERSION = "corp_robocasa_sr_v9"
PROTOCOL_KEYS = {
    "version",
    "checkpoint_sha256",
    "shift",
    "blur_sigma",
    "layouts",
    "layout_id",
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
        or type(protocol["layout_id"]) is not int
        or type(protocol["base_seed"]) is not int
        or protocol["precision"] not in {"fp32", "bf16"}
    ):
        raise ValueError("invalid blur/layout protocol")
    if protocol["layouts"] != sorted(set(protocol["layouts"])):
        raise ValueError("layouts must be sorted and unique")
    if protocol["layout_id"] not in protocol["layouts"]:
        raise ValueError("fixed evaluation layout is outside the allowed layout group")
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
    }
    if (
        set(origin) != expected
        or type(origin["seed"]) is not int
        or type(origin["data_length"]) is not int
        or origin["data_length"] <= 0
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
