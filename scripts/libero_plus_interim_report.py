"""Interim per-dimension report for the full LIBERO-Plus run.

Uses the LIBERO-Plus suite order (the same order the evaluator uses for
`--tasks`) to map task ids to perturbation categories; the category table is
read from the official task_classification.json.
"""

from __future__ import annotations

import contextlib
import glob
import io
import json
import sys
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402

sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import configure_libero  # noqa: E402

VALIDATION = Path("/hy-tmp/validation")
CLASSIFICATION = Path("/hy-tmp/libero_plus/libero/libero/benchmark/task_classification.json")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def load(mode: str, suite: str) -> dict[tuple[int, int], bool]:
    rows = {}
    for path in glob.glob(str(VALIDATION / f"lpfull_{suite}_{mode}_c*" / f"{suite}.jsonl")):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                rows[(int(record["task_id"]), int(record["episode_id"]))] = bool(record["success"])
    return rows


def main() -> int:
    class Args:
        libero_src = Path("/hy-tmp/libero_plus")
        libero_config_path = Path("/hy-tmp/libero_plus_config")
        render_gpu = 0

    with contextlib.redirect_stdout(io.StringIO()):
        configure_libero(Args())
        from libero.libero import benchmark

    classification = json.loads(CLASSIFICATION.read_text())
    buckets: dict[str, dict[str, dict]] = {}
    total_base = total_auto = 0
    for suite_name in SUITES:
        with contextlib.redirect_stdout(io.StringIO()):
            suite = benchmark.get_benchmark_dict()[suite_name]()
        names = suite.get_task_names()
        category_by_name = {item["name"]: item["category"] for item in classification[suite_name]}
        base = load("base", suite_name)
        auto = load("auto", suite_name)
        for key in sorted(set(base) & set(auto)):
            name = names[key[0]]
            category = category_by_name.get(name)
            if category is None:
                continue
            for bucket in (category, "ALL"):
                buckets.setdefault(bucket, {"base": {}, "auto": {}})
                buckets[bucket]["base"][(suite_name,) + key] = base[key]
                buckets[bucket]["auto"][(suite_name,) + key] = auto[key]
        total_base += len(base)
        total_auto += len(auto)
    print(f"episodes: base={total_base} auto={total_auto}")
    print(f"{'dimension':22s} {'n':>5s} {'base':>7s} {'auto':>7s} {'delta':>7s} {'p':>8s}")
    for bucket in sorted(buckets, key=lambda b: (b == "ALL", b)):
        keys = sorted(set(buckets[bucket]["base"]) & set(buckets[bucket]["auto"]))
        n = len(keys)
        only_base = sum(buckets[bucket]["base"][k] and not buckets[bucket]["auto"][k] for k in keys)
        only_auto = sum(buckets[bucket]["auto"][k] and not buckets[bucket]["base"][k] for k in keys)
        p = binomtest(only_base, only_base + only_auto, 0.5).pvalue if (only_base + only_auto) else 1.0
        sr_base = sum(buckets[bucket]["base"][k] for k in keys) / n
        sr_auto = sum(buckets[bucket]["auto"][k] for k in keys) / n
        print(
            f"{bucket:22s} {n:5d} {sr_base * 100:6.1f}% {sr_auto * 100:6.1f}% "
            f"{(sr_auto - sr_base) * 100:+6.1f}% {p:8.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
