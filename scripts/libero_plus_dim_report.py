"""Per-perturbation-dimension report for LIBERO-Plus baseline vs adaptation.

Maps task ids back to LIBERO-Plus categories with task_classification.json and
reports paired statistics per category (and overall).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    ap.add_argument("--combined", type=Path, default=Path("/hy-tmp/validation/lp_combined"))
    ap.add_argument("--classification", type=Path,
                    default=Path("/hy-tmp/libero_plus/libero/libero/benchmark/task_classification.json"))
    ap.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    per_suite: dict[str, dict[int, str]] = {}
    for suite in args.suites:
        meta = json.loads(args.classification.read_text())[suite]
        names = {item["name"]: item for item in meta}
        per_suite[suite] = names

    buckets: dict[str, dict[str, dict]] = defaultdict(lambda: {"base": {}, "auto": {}})
    for suite in args.suites:
        base = load(args.combined / f"{suite}_base.jsonl")
        auto = load(args.combined / f"{suite}_auto.jsonl")
        for key in sorted(set(base) & set(auto)):
            # map (task_id, episode) -> category via the saved selection manifest
            manifest = json.loads(
                Path(f"/hy-tmp/validation/lp_selected_{suite}.json").read_text()
            )
            by_id = {t["task_id"]: t for t in manifest["tasks"]}
            item = by_id.get(key[0])
            if item is None:
                continue
            for bucket in (item["category"], "ALL"):
                buckets[bucket]["base"][(suite,) + key] = base[key]
                buckets[bucket]["auto"][(suite,) + key] = auto[key]

    report = {}
    for bucket, data in sorted(buckets.items()):
        keys = sorted(set(data["base"]) & set(data["auto"]))
        n = len(keys)
        if n == 0:
            continue
        only_base = sum(data["base"][k] and not data["auto"][k] for k in keys)
        only_auto = sum(data["auto"][k] and not data["base"][k] for k in keys)
        p = binomtest(only_base, only_base + only_auto, 0.5).pvalue if (only_base + only_auto) else 1.0
        sr_base = sum(data["base"][k] for k in keys) / n
        sr_auto = sum(data["auto"][k] for k in keys) / n
        report[bucket] = {
            "paired_episodes": n,
            "sr_base": sr_base,
            "sr_auto": sr_auto,
            "delta": sr_auto - sr_base,
            "base_success_auto_fail": only_base,
            "base_fail_auto_success": only_auto,
            "mcnemar_exact_p": float(p),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"{'dimension':22s} {'n':>4s} {'base':>7s} {'auto':>7s} {'delta':>7s} {'p':>9s}")
    for bucket in sorted(report, key=lambda b: (b == "ALL", b)):
        r = report[bucket]
        print(f"{bucket:22s} {r['paired_episodes']:4d} {r['sr_base']*100:6.1f}% {r['sr_auto']*100:6.1f}% "
              f"{r['delta']*100:+6.1f}% {r['mcnemar_exact_p']:9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
