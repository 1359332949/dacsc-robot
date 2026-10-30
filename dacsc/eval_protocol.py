"""Experiment identity for SR aggregation (DomainBed-style grouping by arguments)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from dacsc.corp_images import image_contract
from dacsc.upstream import verify_upstream

PROTOCOL_VERSION = "corp_robocasa_sr_v3"
PROTOCOL_KEYS = {
    "version",
    "checkpoint_sha256",
    "shift",
    "blur_sigma",
    "layouts",
    "base_seed",
    "action_steps",
    "action_horizon",
    "euler_steps",
    "image_size",
    "precision",
    "horizon_source",
    "images",
    "upstream",
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
    if any(
        type(protocol[key]) is not int or protocol[key] <= 0
        for key in ("action_steps", "action_horizon", "euler_steps", "image_size")
    ):
        raise ValueError("invalid inference budget")
    if protocol["action_steps"] > protocol["action_horizon"]:
        raise ValueError("action chunk execution exceeds the prediction horizon")
