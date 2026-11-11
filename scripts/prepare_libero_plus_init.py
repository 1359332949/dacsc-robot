"""Materialise LIBERO-Plus initial states for the released-CoRP evaluator.

LIBERO-Plus resolves perturbed-task initial states through
`Benchmark.get_task_init_states(i)` (suffix-aware), while the official CoRP
evaluation loop loads `init_files/<problem>/<task>.pruned_init` directly. This
script calls the LIBERO-Plus API for the selected tasks and writes the resulting
states to exactly those paths, so the untouched official evaluator runs the
benchmark tasks without any code change. Selection mirrors
scripts/libero_shift_eval.py (`--per-category`, `--categories`, `--seed`).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import configure_libero  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--per-category", type=int, required=True)
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config-path", type=Path, default=Path("/hy-tmp/libero_plus_config"))
    ap.add_argument("--selected-json", type=Path, required=True)
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    configure_libero(args)
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[args.suite]()
    names = suite.get_task_names()
    meta = json.loads(args.classification.read_text())[args.suite]
    groups: dict[str, list] = {}
    for item in meta:
        if args.categories and item["category"] not in args.categories:
            continue
        groups.setdefault(item["category"], []).append(item)
    if not groups:
        raise ValueError("no tasks selected")
    rng = random.Random(args.seed)
    selected = []
    for category in sorted(groups):
        items = groups[category]
        if len(items) < args.per_category:
            raise ValueError(f"insufficient tasks in {category}")
        selected.extend((category, item) for item in rng.sample(items, args.per_category))

    init_root = Path(get_libero_path("init_states"))
    written, skipped = 0, 0
    manifest = []
    for category, item in selected:
        name = item["name"]
        if name not in names:
            raise ValueError(f"task absent from suite: {name}")
        index = names.index(name)
        task = suite.get_task(index)
        target = init_root / task.problem_folder / f"{name}.pruned_init"
        manifest.append(
            {
                "category": category,
                "difficulty_level": item.get("difficulty_level"),
                "name": name,
                "task_id": index,
                "init_path": str(target),
            }
        )
        if target.exists():
            skipped += 1
            continue
        if args.dry_run:
            continue
        states = suite.get_task_init_states(index)
        states = torch.as_tensor(states, dtype=torch.float32)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(states, target)
            written += 1
    args.selected_json.parent.mkdir(parents=True, exist_ok=True)
    args.selected_json.write_text(
        json.dumps(
            {
                "suite": args.suite,
                "seed": args.seed,
                "per_category": args.per_category,
                "categories": args.categories,
                "written": written,
                "skipped": skipped,
                "tasks": manifest,
            },
            indent=2,
        )
    )
    print(json.dumps({"written": written, "skipped": skipped, "selected": len(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
