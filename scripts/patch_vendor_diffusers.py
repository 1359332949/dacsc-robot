#!/usr/bin/env python3
"""Patch vendored robomimic for diffusers >= 0.41 (idempotent)."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "repos/robomimic/robomimic/utils/torch_utils.py"
OLD = """    from diffusers.optimization import (
        Union, SchedulerType, Optional,
        Optimizer, TYPE_TO_SCHEDULER_FUNCTION
    )"""
NEW = """    # diffusers >= 0.41 removed the typing aliases that older robomimic
    # imported here. This function only needs the scheduler enum and map.
    from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION"""


def main():
    s = TARGET.read_text()
    if NEW in s:
        print("already patched")
        return 0
    if OLD not in s:
        raise SystemExit(f"pattern not found in {TARGET}")
    TARGET.write_text(s.replace(OLD, NEW))
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
