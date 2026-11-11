"""Merge LIBERO-Plus evaluation chunks (a/b/r0-r7) into one deduplicated run.

Chunks are produced by scripts/libero_shift_eval.py when the task subset is
split for throughput. Records are deduplicated by (task_id, episode_id); the
first occurrence wins (same seed and same official evaluator for every chunk).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation-root", type=Path, default=Path("/hy-tmp/validation"))
    ap.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    ap.add_argument("--modes", nargs="+", default=["base", "auto"])
    ap.add_argument("--out-dir", type=Path, default=Path("/hy-tmp/validation/lp_combined"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for suite in args.suites:
        for mode in args.modes:
            records: dict[tuple[int, int], dict] = {}
            files = sorted(
                glob.glob(str(args.validation_root / f"lp_{suite}_{mode}_*" / f"{suite}.jsonl"))
            )
            for path in files:
                for line in Path(path).read_text().splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    records.setdefault((int(record["task_id"]), int(record["episode_id"])), record)
            ordered = [records[key] for key in sorted(records)]
            out = args.out_dir / f"{suite}_{mode}.jsonl"
            out.write_text("".join(json.dumps(record) + "\n" for record in ordered))
            successes = sum(record["success"] for record in ordered)
            summary[f"{suite}_{mode}"] = {
                "episodes": len(ordered),
                "successes": successes,
                "success_rate": successes / len(ordered) if ordered else 0.0,
                "chunks": len(files),
            }
            print(f"{suite}_{mode}: {len(ordered)} eps, {successes} success ({len(files)} chunks)")
    (args.out_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
