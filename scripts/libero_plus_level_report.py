"""Per-category and per-difficulty paired report for the full LIBERO-Plus run."""

from __future__ import annotations

import contextlib
import glob
import io
import json
import sys
from collections import defaultdict
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


def load(mode: str) -> dict[tuple[str, int], bool]:
    rows = {}
    for path in glob.glob(str(VALIDATION / f"lpfull_*_{mode}_c*" / "*.jsonl")):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                rows[(record["suite"], int(record["task_id"]))] = bool(record["success"])
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
    keymap = {}
    for suite in SUITES:
        with contextlib.redirect_stdout(io.StringIO()):
            task_suite = benchmark.get_benchmark_dict()[suite]()
        names = task_suite.get_task_names()
        category_by_name = {item["name"]: item["category"] for item in classification[suite]}
        level_by_name = {item["name"]: item.get("difficulty_level") for item in classification[suite]}
        for index, name in enumerate(names):
            keymap[(suite, index)] = (category_by_name.get(name), level_by_name.get(name))

    base = load("base")
    auto = load("auto")
    buckets: dict[tuple[str, str], dict[str, dict]] = defaultdict(lambda: {"b": {}, "a": {}})
    for key in set(base) & set(auto):
        category, level = keymap.get(key, (None, None))
        if category is None:
            continue
        for bucket in (("ALL", "ALL"), (category, "ALL"), ("ALL", str(level)), (category, str(level))):
            buckets[bucket]["b"][key] = base[key]
            buckets[bucket]["a"][key] = auto[key]

    print(f"{'category':22s} {'level':>5s} {'n':>5s} {'base':>7s} {'auto':>7s} {'delta':>7s} {'p':>8s}")
    for bucket in sorted(buckets, key=lambda item: (item[0], item[1])):
        keys = sorted(set(buckets[bucket]["b"]) & set(buckets[bucket]["a"]))
        if not keys:
            continue
        only_base = sum(buckets[bucket]["b"][k] and not buckets[bucket]["a"][k] for k in keys)
        only_auto = sum(buckets[bucket]["a"][k] and not buckets[bucket]["b"][k] for k in keys)
        p = binomtest(only_base, only_base + only_auto, 0.5).pvalue if (only_base + only_auto) else 1.0
        sr_base = sum(buckets[bucket]["b"][k] for k in keys) / len(keys)
        sr_auto = sum(buckets[bucket]["a"][k] for k in keys) / len(keys)
        print(
            f"{bucket[0]:22s} {bucket[1]:>5s} {len(keys):5d} {sr_base * 100:6.1f}% "
            f"{sr_auto * 100:6.1f}% {(sr_auto - sr_base) * 100:+6.1f}% {p:8.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
