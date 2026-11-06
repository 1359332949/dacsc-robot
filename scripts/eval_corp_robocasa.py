"""独立评测 CoRP-RoboCasa checkpoint：per-dim / per-task MSE（高样本量）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dacsc.corp_data import LinearNormalizer  # noqa: E402
from dacsc.corp_robocasa import load_adapted_checkpoint, OBJECTIVE  # noqa: E402
from dacsc.corp_images import SHIFTS, image_contract
from dacsc.eval_protocol import file_sha256
from dacsc.upstream import verify_upstream
from dacsc.run_output import entry_file
from train_corp_robocasa import (  # noqa: E402
    all_task_paths,
    balanced_loader,
    build_dataset,
    evaluate,
)
from scripts.check_data_layout import default_root
from dacsc.action_metrics import MSE_SCHEMA


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-root", type=Path, default=default_root())
    ap.add_argument("--per-task", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--euler-steps", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--train-style-min", type=int, default=11)
    ap.add_argument("--train-style-max", type=int, default=50)
    ap.add_argument("--val-style-min", type=int, default=51)
    ap.add_argument("--val-style-max", type=int, default=60)
    ap.add_argument("--test-style-min", type=int, default=41)
    ap.add_argument("--test-style-max", type=int, default=50)
    ap.add_argument("--source-holdout-residue", type=int, default=9)
    ap.add_argument("--run-test", action="store_true", help="Final evaluation only")
    ap.add_argument(
        "--shifts", nargs="+", choices=SHIFTS, default=["none", "color_jitter"]
    )
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    if min(args.per_task, args.batch_size, args.euler_steps) <= 0:
        ap.error("support, batch-size and Euler steps must be positive")
    with entry_file(Path(args.out)):
        return run(args)


def run(args) -> int:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_digest = file_sha256(Path(args.model))
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    if args.horizon != model.action_horizon:
        raise ValueError("evaluation horizon differs from the checkpoint")
    norm = LinearNormalizer(mean, std)
    print(f"loaded {args.model}, device={device}", flush=True)

    if args.tasks and args.max_tasks:
        raise ValueError("--tasks and --max-tasks are mutually exclusive")
    if not 0 <= args.source_holdout_residue < 10:
        raise ValueError("source holdout residue must be in 0..9")
    tasks = all_task_paths(args.max_tasks, args.data_root, args.tasks)
    print(f"tasks={len(tasks)}", flush=True)
    result = {
        "schema": MSE_SCHEMA,
        "model": args.model,
        "checkpoint_sha256": checkpoint_digest,
        "images": image_contract(),
        "objective": OBJECTIVE,
        "upstream": verify_upstream(),
        "per_task": args.per_task,
        "euler_steps": args.euler_steps,
        "precision": "bf16" if device.type == "cuda" else "fp32",
        "eval_seed": 123,
        "evaluation_batch_size": args.batch_size,
        "training": model.evaluation_origin,
        "splits": {},
    }

    splits = [
        ("source", (args.train_style_min, args.train_style_max)),
        ("val", (args.val_style_min, args.val_style_max)),
    ]
    if args.run_test:
        splits.append(("test", (args.test_style_min, args.test_style_max)))
    for name, (lo, hi) in splits:
        residues = {args.source_holdout_residue} if name == "source" else None
        ds = build_dataset(
            tasks, (lo, hi), args.horizon, 224, True, episode_residues=residues
        )
        loader = balanced_loader(ds, args.batch_size, args.per_task, seed=123)
        conditions = {}
        for shift in args.shifts:
            ev, dt = evaluate(
                model, loader, norm, device, len(loader), args.euler_steps, shift=shift
            )
            conditions[shift] = {**ev, "elapsed_sec": dt}
            print(
                f"{name}/{shift}: micro_action_mse={ev['micro_action_mse']:.4f} macro_task_action_mse={ev['macro_task_action_mse']:.4f} n={len(loader.dataset)} ({dt:.1f}s)",
                flush=True,
            )
        result["splits"][name] = {
            "support": ds.support(loader.dataset.indices),
            "conditions": conditions,
        }

    if file_sha256(Path(args.model)) != result["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed during evaluation")
    Path(args.out).write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    )
    print("DONE ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
