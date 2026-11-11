"""Paired analysis of two LIBERO evaluation runs (same task/seed/episode).

Both runs use the official per-task seed and the same initial-state index, so
episodes are paired; the script reports discordant counts and an exact McNemar
binomial test. Reads the official per-episode JSONL written by evaluate_suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import binomtest


def load(path: Path) -> dict[tuple[int, int], bool]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows[(int(record["task_id"]), int(record["episode_id"]))] = bool(record["success"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a, b = load(args.a), load(args.b)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("no paired episodes")
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if (only_a + only_b) else 1.0
    payload = {
        "paired_episodes": len(keys),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "sr_a": sum(a[k] for k in keys) / len(keys),
        "sr_b": sum(b[k] for k in keys) / len(keys),
        "a_success_b_fail": only_a,
        "a_fail_b_success": only_b,
        "mcnemar_exact_p": float(p),
        "delta_sr_b_minus_a": (sum(b[k] for k in keys) - sum(a[k] for k in keys)) / len(keys),
        "files": {"a": str(args.a), "b": str(args.b)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
