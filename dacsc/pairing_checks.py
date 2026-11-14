"""Pure, unit-testable checks for V1.1 Clean identity and prompt freezing."""

from __future__ import annotations

import re
from pathlib import Path


def instruction_from_task_name(name: str) -> str:
    """Vanilla LIBERO/CoRP filename-derived prompt: underscores -> spaces."""
    return " ".join(str(name).split("_")).strip()


def normalize_prompt(text: str) -> str:
    out = str(text).strip().lower()
    out = re.sub(r"[^a-z0-9\s]", " ", out)
    return re.sub(r"\s+", " ", out).strip()


def validate_clean_file_identity(task_name: str, bddl_path: Path, init_path: Path) -> dict:
    bddl = Path(bddl_path); init = Path(init_path)
    allowed_init = {f"{task_name}.pruned_init", f"{task_name}.init",
                    f"{task_name}.pruned_init.pkl"}
    checks = {
        "bddl_exists": bddl.is_file(),
        "init_exists": init.is_file(),
        "bddl_stem": bddl.stem == task_name,
        "init_name": init.name in allowed_init,
    }
    return {"checks": checks, "pass": all(checks.values()),
            "bddl_stem": bddl.stem, "init_name": init.name}


def freeze_policy_instruction(clean_policy: str, shift_raw_language: str,
                              clean_task_name: str | None = None,
                              require_vanilla_source: bool = False) -> dict:
    """Freeze one prompt and anchor the vanilla source independently."""
    expected_vanilla = instruction_from_task_name(clean_task_name) if clean_task_name else None
    from dacsc.task_pairing import canonical_instruction
    checks = {
        "canonical_matches_shift": bool(
            canonical_instruction(clean_policy) == canonical_instruction(shift_raw_language)),
        "vanilla_source_matches": ((not require_vanilla_source) or
                                   (expected_vanilla is not None and
                                    normalize_prompt(clean_policy) == normalize_prompt(expected_vanilla))),
    }
    return {"checks": checks, "pass": all(checks.values()),
            "expected_vanilla_instruction": expected_vanilla,
            "clean_policy_instruction": clean_policy,
            "shift_policy_instruction": clean_policy}
