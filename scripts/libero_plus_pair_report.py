"""Paired per-dimension report between two LIBERO-Plus result sets."""

from __future__ import annotations

import argparse
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

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def load(pattern: str, suite: str) -> dict[tuple[int, int], bool]:
    rows = {}
    for path in glob.glob(pattern.format(suite=suite)):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                rows[(int(record["task_id"]), int(record["episode_id"]))] = bool(record["success"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="glob with {suite} placeholder")
    ap.add_argument("--b", required=True, help="glob with {suite} placeholder")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    args = ap.parse_args()

    class Args:
        libero_src = Path("/hy-tmp/libero_plus")
        libero_config_path = Path("/hy-tmp/libero_plus_config")
        render_gpu = 0

    with contextlib.redirect_stdout(io.StringIO()):
        configure_libero(Args())
        from libero.libero import benchmark
    classification = json.loads(
        Path("/hy-tmp/libero_plus/libero/libero/benchmark/task_classification.json").read_text()
    )
    buckets: dict[str, dict[str, dict]] = {}
    for suite in SUITES:
        with contextlib.redirect_stdout(io.StringIO()):
            task_suite = benchmark.get_benchmark_dict()[suite]()
        names = task_suite.get_task_names()
        category_by_name = {item["name"]: item["category"] for item in classification[suite]}
        a = load(args.a, suite)
        b = load(args.b, suite)
        for key in sorted(set(a) & set(b)):
            category = category_by_name.get(names[key[0]])
            if category is None:
                continue
            for bucket in (category, "ALL"):
                buckets.setdefault(bucket, {"a": {}, "b": {}})
                buckets[bucket]["a"][(suite,) + key] = a[key]
                buckets[bucket]["b"][(suite,) + key] = b[key]
    print(f"{'dimension':22s} {'n':>5s} {args.label_a:>8s} {args.label_b:>8s} {'delta':>7s} {'p':>8s}")
    for bucket in sorted(buckets, key=lambda name: (name == "ALL", name)):
        keys = sorted(set(buckets[bucket]["a"]) & set(buckets[bucket]["b"]))
        n = len(keys)
        only_a = sum(buckets[bucket]["a"][k] and not buckets[bucket]["b"][k] for k in keys)
        only_b = sum(buckets[bucket]["b"][k] and not buckets[bucket]["a"][k] for k in keys)
        p = binomtest(only_a, only_a + only_b, 0.5).pvalue if (only_a + only_b) else 1.0
        sr_a = sum(buckets[bucket]["a"][k] for k in keys) / n
        sr_b = sum(buckets[bucket]["b"][k] for k in keys) / n
        print(
            f"{bucket:22s} {n:5d} {sr_a * 100:7.1f}% {sr_b * 100:7.1f}% "
            f"{(sr_b - sr_a) * 100:+6.1f}% {p:8.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
