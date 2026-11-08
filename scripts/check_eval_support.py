"""Fail closed when a source panel lacks task-level train support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dacsc.eval_support import validate_source_panel  # noqa: E402
from dacsc.v10_config import load_v10_config  # noqa: E402


def parse_styles(value: str) -> list[int]:
    styles = []
    for part in value.split(","):
        bounds = part.strip().split("-")
        if len(bounds) == 1:
            lo = hi = int(bounds[0])
        elif len(bounds) == 2:
            lo, hi = map(int, bounds)
        else:
            raise ValueError("styles must be integers or ascending ranges")
        if not 1 <= lo <= hi <= 60:
            raise ValueError("styles must be within 1..60")
        styles.extend(range(lo, hi + 1))
    if len(styles) != len(set(styles)):
        raise ValueError("styles must be unique")
    return styles


def main(argv=None) -> int:
    config = load_v10_config()
    train = config["pretrain"]["train_styles"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--styles", type=parse_styles, required=True)
    ap.add_argument("--layout-id", type=int, required=True)
    ap.add_argument("--train-style-min", type=int, default=train[0])
    ap.add_argument("--train-style-max", type=int, default=train[1])
    ap.add_argument(
        "--source-holdout-residue",
        type=int,
        default=config["pretrain"]["source_holdout_residue"],
    )
    ap.add_argument("--require-pairs", action="store_true")
    args = ap.parse_args(argv)
    if len(args.tasks) != len(set(args.tasks)):
        raise ValueError("tasks must be unique")
    tasks = sorted(args.tasks)
    report = validate_source_panel(
        args.data_root.expanduser().resolve(),
        tasks=tasks,
        styles=args.styles,
        layout_id=args.layout_id,
        train_style_range=(args.train_style_min, args.train_style_max),
        source_holdout_residue=args.source_holdout_residue,
        require_pairs=args.require_pairs,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
