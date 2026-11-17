"""Fail-closed TTA split protocol.

A split file is the single source of truth for source/target state roles and
evaluation seeds. The runner must resolve states through this module rather
than reinterpreting matrix heldout indices.
"""

from __future__ import annotations
import hashlib, json
from pathlib import Path

ROLES = {
    "source_fit": "source_train_state_ids",
    "source_validation": "source_val_state_ids",
    "target_dev": "target_dev_state_ids",
    "target_test": "target_test_state_ids",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_split(path: Path) -> dict:
    split = json.loads(Path(path).read_text())
    if split.get("schema") not in {"tta_split_v1", "tta_split_v2"}:
        raise ValueError(f"unsupported split schema: {split.get('schema')}")
    if split.get("status") not in {"frozen", "frozen_pilot_not_confirmatory"}:
        raise ValueError(f"split is not frozen: {split.get('status')}")
    seen = {}
    for role, key in ROLES.items():
        ids = [int(x) for x in split.get(key, [])]
        if not ids:
            raise ValueError(f"split role {role} is empty")
        if len(ids) != len(set(ids)):
            raise ValueError(f"split role {role} has duplicates")
        seen[role] = ids
    flatten = [x for ids in seen.values() for x in ids]
    if len(flatten) != len(set(flatten)):
        raise ValueError("split roles overlap")
    for key in ("evaluation_seeds", "screening_seeds"):
        seeds = [int(x) for x in split.get(key, [])]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError(f"split {key} must be non-empty and unique")
    return split


def role_ids(split: dict, role: str) -> list[int]:
    if role not in ROLES:
        raise ValueError(f"unknown role: {role}")
    return [int(x) for x in split[ROLES[role]]]


def resolve_indices(split: dict, role: str, requested: str, allow_role_override: bool = False) -> list[int]:
    allowed = set(role_ids(split, role))
    if requested.strip():
        ids = [int(x) for x in requested.split(",") if x.strip()]
        extra = set(ids) - allowed
        if extra and not allow_role_override:
            raise ValueError(f"states outside split role {role}: {sorted(extra)}")
        return ids
    return sorted(allowed)


def validate_split_identity(split: dict, suite: str, task: str, lighting_id: int) -> None:
    if split.get("task_suite") != suite:
        raise ValueError(f"split suite {split.get('task_suite')} != {suite}")
    if task and split.get("task") and split.get("task") != task:
        raise ValueError(f"split task {split.get('task')} != {task}")
    if int(split.get("lighting_id", -1)) != int(lighting_id):
        raise ValueError(f"split lighting {split.get('lighting_id')} != {lighting_id}")


def role_for_state(split: dict, state_id: int) -> str | None:
    for role in ROLES:
        if state_id in set(role_ids(split, role)):
            return role
    return None
