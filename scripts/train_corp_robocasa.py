"""CoRP-RoboCasa 无门控 baseline（P2）。

split 由显式 task 列表与 train/val style range 冻结；v10 pilot 使用
train styles 11-50 / internal val 51-60 / official target split。
模型：官方 CoRP 架构 + corp_libero.pt 共享权重，接口层随机初始化。
训练：rectified-flow velocity loss；动作 identity 归一化（对齐 DP 基线）。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from dacsc.corp_data import (  # noqa: E402
    CorpRoboCasaDataset,
    LinearNormalizer,
    collate_corp,
)
from dacsc.corp_images import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    apply_image_shift,
    image_contract,
)
from dacsc.corp_robocasa import (  # noqa: E402
    CHECKPOINT_FORMAT,
    OBJECTIVE,
    ROBOCASA_CORP_CONFIG,
    TASK_AUX_HEAD_HIDDEN_DIM,
    TASK_AUX_PROVENANCE,
    CorpRoboCasa,
    TaskAuxConfig,
    load_corp_robocasa,
    _CORP_REPO,
)
from dacsc.training_state import (  # noqa: E402
    ModelEMA,
    atomic_torch_save,
    evaluation_state,
    restore_training_state,
    training_state,
)
from dacsc.run_output import entry_output  # noqa: E402
from dacsc.upstream import verify_upstream, verify_release_checkpoint
from scripts.check_data_layout import default_root
from dacsc.action_metrics import MSE_SCHEMA, action_mse_summary
from dacsc.pixel_augmentation import (
    TRAIN_AUGMENTATIONS,
    TrainingPixelAugmentation,
    pixel_augmentation_contract,
)
from dacsc.code_identity import verify_package_identity

DATA_ROOT = default_root()
CKPT = _CORP_REPO / "checkpoints/corp_libero.pt"


def all_task_paths(
    max_tasks: int = 0,
    data_root: Path = DATA_ROOT,
    task_names: list[str] | None = None,
) -> list[str]:
    if max_tasks and task_names:
        raise ValueError("--max-tasks and --tasks are mutually exclusive")
    paths = []
    for task_dir in sorted((data_root / "v1.0/pretrain/atomic").iterdir()):
        if not task_dir.is_dir():
            continue
        cand = sorted(task_dir.glob("*/lerobot"))
        if len(cand) != 1:
            raise ValueError(f"expected one dataset release per task: {task_dir}")
        paths.append(str(cand[0]))
    if not paths:
        raise ValueError("no task datasets found")
    if task_names:
        if len(task_names) != len(set(task_names)):
            raise ValueError("--tasks contains duplicates")
        by_name = {Path(path).parent.parent.name: path for path in paths}
        missing = sorted(set(task_names) - set(by_name))
        if missing:
            raise ValueError(f"unknown tasks: {missing}")
        paths = [by_name[name] for name in sorted(task_names)]
    if max_tasks:
        paths = paths[:max_tasks]
    return paths


def project_code_contract() -> dict:
    def git(*args: str):
        return subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    revision = git("rev-parse", "HEAD")
    if revision.returncode != 0:
        return verify_package_identity(_REPO_ROOT)
    dirty = git("diff", "--quiet").returncode != 0 or git(
        "diff", "--cached", "--quiet"
    ).returncode != 0
    return {"git_commit": revision.stdout.strip(), "git_dirty": dirty}


def data_contract(train_ds, source_ds, val_ds, tasks, args) -> dict:
    def compact(dataset):
        support = dataset.support()
        return {
            "frames": support["frames"],
            "episodes": support["episodes"],
            "metadata_sha256": support["metadata_sha256"],
            "selection_sha256": support["selection_sha256"],
        }

    task_names = sorted(Path(path).parent.parent.name for path in tasks)
    code = project_code_contract()
    return {
        "tasks": task_names,
        "train_styles": [args.train_style_min, args.train_style_max],
        "val_styles": [args.val_style_min, args.val_style_max],
        "source_holdout_residue": args.source_holdout_residue,
        "train": compact(train_ds),
        "source": compact(source_ds),
        "val": compact(val_ds),
        "project_git_commit": code["git_commit"],
        "project_git_dirty": code["git_dirty"],
    }


def task_class_map(tasks) -> dict[str, int]:
    """Stable task-name -> class-id mapping shared by train/source/val datasets."""
    names = [Path(path).parent.parent.name for path in tasks]
    if len(names) != len(set(names)):
        raise ValueError("task class map contains duplicate task names")
    return {name: index for index, name in enumerate(sorted(names))}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataset(
    tasks,
    split,
    horizon,
    image_size,
    use_cache,
    episode_residues=None,
    *,
    normalize_images=True,
    cache_videos=False,
):
    lo, hi = split
    return CorpRoboCasaDataset(
        tasks,
        style_min=lo,
        style_max=hi,
        horizon=horizon,
        image_size=image_size,
        use_cache=use_cache,
        episode_residues=episode_residues,
        normalize_images=normalize_images,
        allow_raw_rgb=not normalize_images,
        cache_videos=cache_videos,
    )


def move_batch(
    batch,
    norm,
    device,
    *,
    pixel_augmentation=None,
    task_to_id: dict[str, int] | None = None,
):
    images = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
    if batch["image_space"] == "rgb01":
        if pixel_augmentation is None:
            raise ValueError(
                "raw RGB cannot enter evaluation or training without an explicit training augmentation"
            )
        views = list(images)
        augmented = pixel_augmentation(torch.stack([images[v] for v in views], dim=1))
        images = {v: augmented[:, i] for i, v in enumerate(views)}
    elif (
        batch["image_space"] != "imagenet_normalized" or pixel_augmentation is not None
    ):
        raise ValueError(
            "pixel training augmentation requires raw RGB, not normalized/evaluation images"
        )
    moved = {
        "images": images,
        "state": norm.normalize_state(batch["state"].to(device)),
        "instruction": batch["instruction"],
        "action": batch["action"].to(device),
        "action_mask": batch["action_mask"].to(device),
        "style_id": batch["style_id"].to(device),
    }
    if task_to_id is not None:
        try:
            task_ids = [task_to_id[name] for name in batch["task"]]
        except KeyError as exc:
            raise ValueError(f"task has no class id: {exc.args[0]}") from exc
        moved["task_id"] = torch.tensor(task_ids, dtype=torch.long, device=device)
    return moved


def balanced_indices(dataset, per_task: int, seed: int = 0) -> list[int]:
    """每个 task 取 per_task 帧，并在该 task 内部的各 style 间尽量均分。

    按 (task, style) 分层，避免 per-task gap 被 style 采样偏差污染；
    当 style 数多于 per_task 时，每个 style 至少 1 帧（总数可能略超 per_task）。
    """
    by_task: dict[str, dict[int, list[int]]] = {}
    for i, entry in enumerate(dataset.flat):
        by_task.setdefault(entry[4], {}).setdefault(entry[2], []).append(i)
    rng = random.Random(seed)
    out: list[int] = []
    for task in sorted(by_task):
        styles = sorted(by_task[task])
        per_style = max(1, per_task // len(styles))
        for style in styles:
            idx = by_task[task][style][:]
            rng.shuffle(idx)
            out.extend(idx[:per_style])
    return out


def balanced_loader(dataset, batch_size, per_task, seed, num_workers=0):
    subset = Subset(dataset, balanced_indices(dataset, per_task, seed))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_corp,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    norm,
    device,
    max_batches,
    euler_steps,
    shift: str = "none",
    *,
    ema=None,
    eval_seed: int = 123,
):
    with evaluation_state(model, ema, seed=eval_seed):
        return _evaluate(model, loader, norm, device, max_batches, euler_steps, shift)


@torch.no_grad()
def _evaluate(model, loader, norm, device, max_batches, euler_steps, shift):
    """Return standard masked scalar, task-macro, group and twelve-axis MSE.

    action 维度顺序（与 shape_meta 的 lerobot_keys 拼接顺序一致）：
        eef_position(0:3) + eef_rotation(3:6) + gripper_close(6:7)
        + base_motion(7:11) + control_mode(11:12)
    """
    model.eval()
    records = []
    horizon = None
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        b = move_batch(batch, norm, device)
        images = b["images"]
        if shift != "none":
            mean = IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
            std = IMAGENET_STD.to(device).view(1, 3, 1, 1)
            images = {
                k: (apply_image_shift(v * std + mean, shift) - mean) / std
                for k, v in images.items()
            }
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            pred = model.sample_action(
                images,
                b["state"],
                b["instruction"],
                num_steps=euler_steps,
            )
        pred = norm.unnormalize_action(pred.float())
        target = b["action"].float()
        if pred.shape != target.shape:
            raise ValueError(
                "prediction/target shape mismatch; broadcasting is not a valid evaluation"
            )
        valid = b["action_mask"]
        if (
            valid.dtype != torch.bool
            or valid.shape != target.shape[:2]
            or not valid.any()
        ):
            raise ValueError("invalid evaluation action mask")
        if horizon is not None and horizon != target.shape[1]:
            raise ValueError("mixed prediction horizons in one evaluation")
        horizon = target.shape[1]
        se = F.mse_loss(pred.float(), target.float(), reduction="none")
        if not torch.isfinite(se[valid]).all():
            raise FloatingPointError("nonfinite evaluation actions")
        # One device transfer per batch; masked-out errors cannot enter any sum.
        sums = (
            se.masked_fill(~valid[..., None], 0)
            .sum(dim=1, dtype=torch.float64)
            .cpu()
            .tolist()
        )
        positions = valid.sum(dim=1).cpu().tolist()
        styles = batch["style_id"].tolist()
        for j, task in enumerate(batch["task"]):
            records.append(
                {
                    "task": task,
                    "episode": batch["episode"][j],
                    "frame": batch["frame"][j],
                    "style_id": styles[j],
                    "valid_action_positions": positions[j],
                    "squared_error_sum": sums[j],
                }
            )
    out = action_mse_summary(records)
    out["prediction_horizon"] = horizon
    return out, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Explicit total budget; no implicit smoke run",
    )
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="RoboCasa365 root containing v1.0/",
    )
    ap.add_argument(
        "--checkpoint", type=Path, default=CKPT, help="Official LIBERO initialization"
    )
    ap.add_argument("--per-task-eval", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ema-decay", type=float, default=0.99, help="0 disables EMA")
    ap.add_argument(
        "--task-aux-lambda",
        type=float,
        default=0.01,
        help="CoRP Eq.7 task-classification loss weight; 0 disables the auxiliary head",
    )
    ap.add_argument(
        "--task-aux-depth",
        type=int,
        default=1,
        help="DiT prefix depth used by the shared task head; paper default J={1}",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit RoboCasa atomic task names; default is all tasks",
    )
    ap.add_argument("--train-style-min", type=int, default=11)
    ap.add_argument("--train-style-max", type=int, default=50)
    ap.add_argument("--val-style-min", type=int, default=51)
    ap.add_argument("--val-style-max", type=int, default=60)
    ap.add_argument("--source-holdout-residue", type=int, default=9)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument(
        "--save-every", type=int, default=0, help="0 = save only at the end"
    )
    ap.add_argument("--euler-steps", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument(
        "--resume",
        help="Full training checkpoint; --steps remains the original total budget",
    )
    ap.add_argument(
        "--train-augmentation",
        choices=TRAIN_AUGMENTATIONS,
        default="none",
        help="Train-only Torchvision photometric baseline; separate from --eval-shift",
    )
    ap.add_argument(
        "--eval-shift",
        default="color_jitter",
        choices=["none", "grayscale", "color_jitter", "low_contrast", "hue_shift"],
    )
    args = ap.parse_args()
    if args.steps <= 0 or args.eval_every <= 0 or args.save_every < 0:
        ap.error("steps/eval-every must be positive and save-every nonnegative")
    if args.horizon != ROBOCASA_CORP_CONFIG["action_horizon"]:
        ap.error("horizon must match the CoRP-RoboCasa architecture")
    if (
        min(args.batch_size, args.eval_batch_size, args.per_task_eval, args.euler_steps)
        <= 0
    ):
        ap.error("batch sizes, per-task support and Euler steps must be positive")
    if args.resume and args.num_workers != 0:
        ap.error("exact data-order resume currently requires --num-workers 0")
    if args.tasks and args.max_tasks:
        ap.error("--tasks and --max-tasks are mutually exclusive")
    if (
        not 0 <= args.source_holdout_residue < 10
        or not 11 <= args.train_style_min <= args.train_style_max <= 60
        or not 11 <= args.val_style_min <= args.val_style_max <= 60
    ):
        ap.error("invalid source residue or style range")
    train_styles = set(range(args.train_style_min, args.train_style_max + 1))
    val_styles = set(range(args.val_style_min, args.val_style_max + 1))
    if train_styles & val_styles:
        ap.error("train and val style ranges must be disjoint")
    if (
        args.num_workers < 0
        or args.max_tasks < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or not math.isfinite(args.task_aux_lambda)
        or args.task_aux_lambda < 0
        or args.task_aux_depth < 0
        or args.task_aux_depth > ROBOCASA_CORP_CONFIG["action_generator"]["num_layers"]
        or not 0 <= args.ema_decay < 1
    ):
        ap.error("invalid workers, task limit, LR or EMA decay")

    with entry_output(Path(args.out_dir), resume=bool(args.resume)) as out:
        return run(args, out)


def run(args, out: Path) -> int:
    upstream = verify_upstream()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "P2 baseline 必须用 GPU"

    tasks = all_task_paths(args.max_tasks, args.data_root, args.tasks)
    task_to_id = task_class_map(tasks)
    task_aux_cfg = (
        None
        if args.task_aux_lambda == 0
        else TaskAuxConfig(
            num_classes=len(task_to_id),
            loss_weight=args.task_aux_lambda,
            depths=(args.task_aux_depth,),
            head_hidden_dim=TASK_AUX_HEAD_HIDDEN_DIM,
        )
    )
    task_aux_payload = (
        None
        if task_aux_cfg is None
        else {
            **task_aux_cfg.to_dict(),
            "task_to_id": task_to_id,
            "implementation": TASK_AUX_PROVENANCE,
        }
    )
    print(f"tasks={len(tasks)}", flush=True)
    print(f"task_aux={task_aux_payload}", flush=True)
    t0 = time.time()
    train_ds = build_dataset(
        tasks,
        (args.train_style_min, args.train_style_max),
        args.horizon,
        224,
        True,
        episode_residues=set(range(10)) - {args.source_holdout_residue},
        normalize_images=args.train_augmentation == "none",
        cache_videos=True,
    )  # 90% source train
    source_ds = build_dataset(
        tasks,
        (args.train_style_min, args.train_style_max),
        args.horizon,
        224,
        False,
        episode_residues={args.source_holdout_residue},
    )  # 10% source holdout
    val_ds = build_dataset(
        tasks,
        (args.val_style_min, args.val_style_max),
        args.horizon,
        224,
        False,
    )
    print(
        f"datasets built in {time.time()-t0:.1f}s: train={len(train_ds)} "
        f"source_val={len(source_ds)} val={len(val_ds)}",
        flush=True,
    )
    selection_contract = data_contract(train_ds, source_ds, val_ds, tasks, args)
    print(f"data_contract={selection_contract}", flush=True)

    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            resume_payload["format"] != CHECKPOINT_FORMAT
            or resume_payload["objective"] != OBJECTIVE
            or resume_payload["images"] != image_contract()
            or resume_payload["upstream"] != upstream
        ):
            raise ValueError("resume training/image contract differs")
        if resume_payload["training_state"]["step"] >= args.steps:
            raise ValueError(
                "training budget is already complete; use the standalone evaluator "
                "instead of resuming or publishing potentially stale metrics"
            )
        if resume_payload.get("metric_schema") != MSE_SCHEMA:
            raise ValueError(
                "resume metric schema differs; legacy metric histories are unsupported"
            )
        if resume_payload.get("task_aux") != task_aux_payload:
            raise ValueError(
                "resume task auxiliary objective/class mapping differs from checkpoint"
            )
        saved_args = resume_payload["run_args"]
        # Reporting/checkpoint cadence and destination may change; training may not.
        mutable = {
            "resume",
            "out_dir",
            "eval_every",
            "save_every",
            "eval_batch_size",
            "per_task_eval",
        }
        for key, value in vars(args).items():
            if isinstance(value, Path):
                value = str(value)
            if key not in mutable and saved_args[key] != value:
                raise ValueError(f"resume argument mismatch: {key}")
        norm = LinearNormalizer(**resume_payload["normalizer"])
        if resume_payload["config"] != ROBOCASA_CORP_CONFIG:
            raise ValueError("resume architecture differs from the configured model")
    else:
        norm = LinearNormalizer.fit(train_ds, max_steps=512)
        print("normalizer fitted on train styles", flush=True)

    pixel_config = pixel_augmentation_contract(args.train_augmentation, args.seed)
    pixel_augmentation = (
        TrainingPixelAugmentation(args.seed) if pixel_config is not None else None
    )
    if (
        resume_payload is not None
        and resume_payload["pixel_augmentation"] != pixel_config
    ):
        raise ValueError("resume training pixel augmentation differs from checkpoint")

    if resume_payload is None:
        initialization_sha256 = verify_release_checkpoint(args.checkpoint)
        model, merge = load_corp_robocasa(
            args.checkpoint,
            config=ROBOCASA_CORP_CONFIG,
            device=device,
            task_aux=task_aux_cfg,
        )
    else:
        initialization_sha256 = resume_payload["initialization_sha256"]
        model = CorpRoboCasa(
            resume_payload["config"],
            task_aux=task_aux_cfg,
        ).to(device)
        merge = resume_payload["merge"]
    model.train()
    print(
        f"model loaded: copied={merge['copied']} random={merge['random_init']}",
        flush=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    warmup_steps = max(1, args.steps // 10)
    optimization_recipe = {
        "optimizer": "torch.optim.AdamW",
        "betas": opt.defaults["betas"],
        "eps": opt.defaults["eps"],
        "weight_decay": opt.defaults["weight_decay"],
        "scheduler": "torch.optim.lr_scheduler.LambdaLR: linear warmup then constant",
        "warmup_steps": warmup_steps,
        "provenance": "explicit project adaptation; not an official CoRP training recipe",
    }
    training_origin_base = {
        "seed": args.seed,
        "budget": {
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "ema_decay": args.ema_decay,
            "grad_clip": 1.0,
            "precision": "bf16",
            "n_obs_steps": 1,
        },
        "optimization": optimization_recipe,
        "pixel_augmentation": pixel_config,
        "task_aux": task_aux_payload,
        "initialization_policy": merge["initialization_policy"],
        "initialization_sha256": initialization_sha256,
        "data_metadata_sha256": train_ds.support()["metadata_sha256"],
        "data_length": len(train_ds),
        "data_contract": selection_contract,
    }
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    def save_checkpoint(path, completed_step):
        training_origin = {
            **training_origin_base,
            "completed_steps": completed_step,
        }
        atomic_torch_save(
            {
                "format": CHECKPOINT_FORMAT,
                "metric_schema": MSE_SCHEMA,
                "objective": OBJECTIVE,
                "images": image_contract(),
                "upstream": upstream,
                "initialization_sha256": initialization_sha256,
                "initialization_policy": merge["initialization_policy"],
                "optimization": optimization_recipe,
                "config": ROBOCASA_CORP_CONFIG,
                "model": ema.shadow if ema is not None else model.state_dict(),
                "raw_model": model.state_dict(),
                "budget": {
                    "batch_size": args.batch_size,
                    "steps": args.steps,
                    "lr": args.lr,
                    "ema_decay": args.ema_decay,
                    "grad_clip": 1.0,
                    "precision": "bf16",
                    "n_obs_steps": 1,
                },
                "normalizer": {
                    "state_mean": norm.state_mean.cpu(),
                    "state_std": norm.state_std.cpu(),
                },
                "task_aux": task_aux_payload,
                "pixel_augmentation": pixel_config,
                "merge": merge,
                "training_origin": training_origin,
                "run_args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "metrics": metrics,
                "training_state": training_state(
                    opt,
                    sched,
                    ema,
                    step=completed_step,
                    data_order={
                        "epoch_rng": epoch_rng,
                        "batches_in_epoch": batches_in_epoch,
                        "dataset_paths": tasks,
                        "dataset_length": len(train_ds),
                        "metadata_sha256": train_ds.support()["metadata_sha256"],
                        "data_contract": selection_contract,
                    },
                    augmentation=pixel_augmentation,
                ),
            },
            path,
        )

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup_steps)
    )

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_corp,
        pin_memory=True,
        drop_last=True,
        generator=loader_generator,
    )
    # 评测用 per-task 均衡采样，保证 61 个 task 都被覆盖。
    val_loader = balanced_loader(val_ds, args.eval_batch_size, args.per_task_eval, 123)
    source_loader = balanced_loader(
        source_ds, args.eval_batch_size, args.per_task_eval, 123
    )

    metrics = [] if resume_payload is None else resume_payload["metrics"]
    if resume_payload is not None:
        saved_order = resume_payload["training_state"]["data_order"]
        if saved_order["dataset_paths"] != tasks or saved_order[
            "dataset_length"
        ] != len(train_ds):
            raise ValueError("resume dataset paths/length differ from the checkpoint")
        if saved_order["metadata_sha256"] != train_ds.support()["metadata_sha256"]:
            raise ValueError("resume episode metadata differs from the checkpoint")
        if saved_order.get("data_contract") != selection_contract:
            raise ValueError("resume data selection contract differs from the checkpoint")
        loader_generator.set_state(saved_order["epoch_rng"])
    epoch_rng = loader_generator.get_state()
    it = iter(train_loader)
    batches_in_epoch = 0
    step = 0
    if resume_payload is not None:
        batches_in_epoch = resume_payload["training_state"]["data_order"][
            "batches_in_epoch"
        ]
        for _ in range(batches_in_epoch):
            next(it)
        step = restore_training_state(
            resume_payload, model, opt, sched, ema, augmentation=pixel_augmentation
        )
        if step > args.steps:
            raise ValueError("resume checkpoint exceeds the total step budget")
        print(f"resumed at completed step {step}", flush=True)
    t_start = time.time()
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            epoch_rng = loader_generator.get_state()
            it = iter(train_loader)
            batches_in_epoch = 0
            batch = next(it)
        batches_in_epoch += 1
        b = move_batch(
            batch,
            norm,
            device,
            pixel_augmentation=pixel_augmentation,
            task_to_id=task_to_id,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model.training_loss(
                b["images"],
                b["state"],
                b["instruction"],
                b["action"],
                b["action_mask"],
                task_labels=b["task_id"] if task_aux_cfg is not None else None,
            )
        loss = losses.total
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
        if ema is not None:
            ema.update(model)
        sched.step()
        if step % 5 == 0 or step == args.steps - 1:
            task_loss = (
                " n/a" if losses.task is None else f" task {float(losses.task):.4f}"
            )
            print(
                f"step {step:4d} loss {float(loss):.4f} "
                f"flow {float(losses.flow):.4f}{task_loss} "
                f"lr {sched.get_last_lr()[0]:.2e} elapsed {time.time()-t_start:.1f}s",
                flush=True,
            )
        step += 1
        # Saving is independent of evaluation cadence and records the actual step.
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(out / f"model_step{step}.pt", step)
            print(f"  saved model_step{step}.pt", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            src_eval, dt1 = evaluate(
                model,
                source_loader,
                norm,
                device,
                len(source_loader),
                args.euler_steps,
                ema=ema,
            )
            val_eval, dt2 = evaluate(
                model,
                val_loader,
                norm,
                device,
                len(val_loader),
                args.euler_steps,
                ema=ema,
            )
            src_mse, val_mse = (
                src_eval["micro_action_mse"],
                val_eval["micro_action_mse"],
            )
            print(
                f"  step {step}: source_mse={src_mse:.4f} val_mse={val_mse:.4f} "
                f"gap={val_mse-src_mse:+.4f} ({dt1+dt2:.1f}s)",
                flush=True,
            )
            src_shift_eval, dt3 = evaluate(
                model,
                source_loader,
                norm,
                device,
                len(source_loader),
                args.euler_steps,
                shift=args.eval_shift,
                ema=ema,
            )
            val_shift_eval, dt4 = evaluate(
                model,
                val_loader,
                norm,
                device,
                len(val_loader),
                args.euler_steps,
                shift=args.eval_shift,
                ema=ema,
            )
            print(
                f"    shifted({args.eval_shift}): "
                f"source={src_shift_eval['micro_action_mse']:.4f} "
                f"val={val_shift_eval['micro_action_mse']:.4f} ({dt3+dt4:.1f}s)",
                flush=True,
            )
            print(
                "    per-action-group source:",
                {k: round(v, 4) for k, v in src_eval["per_action_group_mse"].items()},
                flush=True,
            )
            print(
                "    per-action-group val   :",
                {k: round(v, 4) for k, v in val_eval["per_action_group_mse"].items()},
                flush=True,
            )
            # Keep frame/episode detail on disk, not duplicated in every checkpoint.
            detail = {
                "schema": MSE_SCHEMA,
                "scope": "training_diagnostic",
                "step": step,
                "source": {"clean": src_eval, "shifted": src_shift_eval},
                "val": {"clean": val_eval, "shifted": val_shift_eval},
                "shift": args.eval_shift,
                "eval_seed": 123,
            }
            metrics_path = out / f"action_metrics_step{step}.json"
            tmp = metrics_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(detail, allow_nan=False))
            tmp.replace(metrics_path)
            from dacsc.eval_protocol import file_sha256

            metrics.append(
                {
                    "step": step,
                    "schema": MSE_SCHEMA,
                    "eval_shift": args.eval_shift,
                    "source": {
                        k: v
                        for k, v in src_eval.items()
                        if k not in ("records", "per_episode")
                    },
                    "val": {
                        k: v
                        for k, v in val_eval.items()
                        if k not in ("records", "per_episode")
                    },
                    "source_shifted": {
                        k: v
                        for k, v in src_shift_eval.items()
                        if k not in ("records", "per_episode")
                    },
                    "val_shifted": {
                        k: v
                        for k, v in val_shift_eval.items()
                        if k not in ("records", "per_episode")
                    },
                    "val_minus_source_micro_action_mse": val_mse - src_mse,
                    "val_minus_source_macro_task_action_mse": val_eval[
                        "macro_task_action_mse"
                    ]
                    - src_eval["macro_task_action_mse"],
                    "detail_file": metrics_path.name,
                    "detail_sha256": file_sha256(metrics_path),
                    "train_total_loss": float(loss),
                    "train_flow_loss": float(losses.flow),
                    "train_task_aux_loss": (
                        None if losses.task is None else float(losses.task)
                    ),
                }
            )
            model.train()

    result = {
        "metric_schema": MSE_SCHEMA,
        "pixel_augmentation": pixel_config,
        "task_aux": task_aux_payload,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "objective": OBJECTIVE,
        "images": image_contract(),
        "upstream": upstream,
        "initialization_sha256": initialization_sha256,
        "initialization_policy": merge["initialization_policy"],
        "optimization": optimization_recipe,
        "training_origin": {
            **training_origin_base,
            "completed_steps": step,
        },
        "data_contract": selection_contract,
        "support": {
            "train": train_ds.support(),
            "source": source_ds.support(),
            "val": val_ds.support(),
            "source_eval": source_ds.support(source_loader.dataset.indices),
            "val_eval": val_ds.support(val_loader.dataset.indices),
        },
        "merge": {k: merge[k] for k in ("copied", "random_init", "partial_keys")},
        "budget": {
            "batch_size": args.batch_size,
            "steps": args.steps,
            "lr": args.lr,
            "ema_decay": args.ema_decay,
            "grad_clip": 1.0,
            "precision": "bf16",
            "n_obs_steps": 1,
        },
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "metrics": metrics,
    }
    save_checkpoint(out / "model.pt", step)
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("DONE ->", out / "result.json", "| model.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
