"""CoRP-RoboCasa 无门控 baseline（P2）。

严格 split：train styles 11-30 / val 31-40 / test 41-50。
模型：官方 CoRP 架构 + corp_libero.pt 共享权重，接口层随机初始化。
训练：rectified-flow velocity loss；动作 identity 归一化（对齐 DP 基线）。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
    CORP_STYLE_MODES,
    OBJECTIVE,
    ROBOCASA_CORP_CONFIG,
    CorpRoboCasa,
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

DATA_ROOT = default_root()
CKPT = _CORP_REPO / "checkpoints/corp_libero.pt"


# action 维度顺序（shape_meta 拼接顺序）。
ACTION_GROUPS: dict[str, slice] = {
    "eef_position": slice(0, 3),
    "eef_rotation": slice(3, 6),
    "gripper_close": slice(6, 7),
    "base_motion": slice(7, 11),
    "control_mode": slice(11, 12),
}


def all_task_paths(max_tasks: int = 0, data_root: Path = DATA_ROOT) -> list[str]:
    paths = []
    for task_dir in sorted((data_root / "v1.0/pretrain/atomic").iterdir()):
        if not task_dir.is_dir():
            continue
        cand = sorted(task_dir.glob("*/lerobot"))
        if len(cand) != 1:
            raise ValueError(f"expected one dataset release per task: {task_dir}")
        paths.append(str(cand[0]))
        if max_tasks and len(paths) >= max_tasks:
            break
    if not paths:
        raise ValueError("no task datasets found")
    return paths


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataset(tasks, split, horizon, image_size, use_cache, episode_residues=None):
    lo, hi = split
    return CorpRoboCasaDataset(
        tasks,
        style_min=lo,
        style_max=hi,
        horizon=horizon,
        image_size=image_size,
        use_cache=use_cache,
        episode_residues=episode_residues,
    )


def move_batch(batch, norm, device):
    return {
        "images": {
            k: v.to(device, non_blocking=True) for k, v in batch["images"].items()
        },
        "state": norm.normalize_state(batch["state"].to(device)),
        "instruction": batch["instruction"],
        "action": batch["action"].to(device),
        "action_mask": batch["action_mask"].to(device),
        "style_id": batch["style_id"].to(device),
    }


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
    """返回 overall / per-dim / per-task MSE。

    action 维度顺序（与 shape_meta 的 lerobot_keys 拼接顺序一致）：
        eef_position(0:3) + eef_rotation(3:6) + gripper_close(6:7)
        + base_motion(7:11) + control_mode(11:12)
    """
    model.eval()
    total, n = 0.0, 0
    group_sum = {g: 0.0 for g in ACTION_GROUPS}
    group_n = {g: 0 for g in ACTION_GROUPS}
    task_sum: dict[str, float] = {}
    task_n: dict[str, int] = {}
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
                domain_ids=b["style_id"],
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
        se = (pred - target) ** 2
        if not torch.isfinite(se[valid]).all():
            raise FloatingPointError("nonfinite evaluation actions")
        total += se[valid].sum().item()
        n += se[valid].numel()
        for g, sl in ACTION_GROUPS.items():
            group_sum[g] += se[..., sl][valid].sum().item()
            group_n[g] += se[..., sl][valid].numel()
        for j, task in enumerate(batch["task"]):
            task_sum[task] = task_sum.get(task, 0.0) + se[j][valid[j]].sum().item()
            task_n[task] = task_n.get(task, 0) + se[j][valid[j]].numel()
    if not n:
        raise ValueError("evaluation produced no valid actions")
    out = {
        "overall": total / max(n, 1),
        "per_dim": {g: group_sum[g] / max(group_n[g], 1) for g in ACTION_GROUPS},
        "per_task": {t: task_sum[t] / max(task_n[t], 1) for t in task_sum},
        "valid_action_scalars": n,
    }
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tasks", type=int, default=0)
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
    ap.add_argument("--run-test", action="store_true")
    ap.add_argument(
        "--token-style-mode",
        default="none",
        choices=CORP_STYLE_MODES,
    )
    ap.add_argument("--token-style-alpha", type=float, default=0.5)
    ap.add_argument("--token-style-action-weight", type=float, default=1.0)
    ap.add_argument("--token-style-protect-ratio", type=float, default=0.0)
    ap.add_argument(
        "--token-style-stats", default="meanstd", choices=["meanstd", "mean", "std"]
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
    if (
        args.num_workers < 0
        or args.max_tasks < 0
        or not math.isfinite(args.lr)
        or args.lr <= 0
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

    tasks = all_task_paths(args.max_tasks, args.data_root)
    print(f"tasks={len(tasks)}", flush=True)
    t0 = time.time()
    train_ds = build_dataset(
        tasks, (11, 30), args.horizon, 224, True, episode_residues=set(range(9))
    )  # 90% source train
    source_ds = build_dataset(
        tasks, (11, 30), args.horizon, 224, True, episode_residues={9}
    )  # 10% source holdout
    val_ds = build_dataset(tasks, (31, 40), args.horizon, 224, True)
    print(
        f"datasets built in {time.time()-t0:.1f}s: train={len(train_ds)} "
        f"source_val={len(source_ds)} val={len(val_ds)}",
        flush=True,
    )

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
        saved_args = resume_payload["run_args"]
        # Reporting/checkpoint cadence and destination may change; training may not.
        mutable = {
            "resume",
            "out_dir",
            "eval_every",
            "save_every",
            "run_test",
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

    token_style_cfg = None
    token_style = None
    if args.token_style_mode != "none":
        from dacsc.token_style import TokenStyleModulator

        token_style_cfg = {
            "mode": args.token_style_mode,
            "alpha": args.token_style_alpha,
            "action_weight": args.token_style_action_weight,
            "protect_ratio": args.token_style_protect_ratio,
            "stats": args.token_style_stats,
            "dim": 384,
            "donor_seed": args.seed,
        }
        token_style = TokenStyleModulator(**token_style_cfg).to(device)
        print(f"token_style enabled: {token_style_cfg}", flush=True)
    if resume_payload is None:
        initialization_sha256 = verify_release_checkpoint(args.checkpoint)
        model, merge = load_corp_robocasa(
            args.checkpoint,
            config=ROBOCASA_CORP_CONFIG,
            device=device,
            token_style=token_style,
        )
    else:
        initialization_sha256 = resume_payload["initialization_sha256"]
        model = CorpRoboCasa(resume_payload["config"], token_style=token_style).to(
            device
        )
        merge = resume_payload["merge"]
    if token_style is not None:
        assert model._style_handle is not None, "token_style hook not registered"
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
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    def save_checkpoint(path, completed_step):
        runtime = model.training_runtime_state()
        atomic_torch_save(
            {
                "format": CHECKPOINT_FORMAT,
                "objective": OBJECTIVE,
                "images": image_contract(),
                "upstream": upstream,
                "initialization_sha256": initialization_sha256,
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
                "token_style": token_style_cfg,
                "merge": merge,
                "training_runtime": runtime,
                "run_args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "metrics": metrics,
                "training_state": training_state(
                    model,
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
                    },
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
        step = restore_training_state(resume_payload, model, opt, sched, ema)
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
        b = move_batch(batch, norm, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.flow_matching_loss(
                b["images"],
                b["state"],
                b["instruction"],
                b["action"],
                b["action_mask"],
                domain_ids=b["style_id"],
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if getattr(model, "token_style", None) is not None:
            model.update_action_importance()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        opt.step()
        if ema is not None:
            ema.update(model)
        sched.step()
        if step % 5 == 0 or step == args.steps - 1:
            print(
                f"step {step:4d} loss {float(loss):.4f} "
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
            src_mse, val_mse = src_eval["overall"], val_eval["overall"]
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
                f"source={src_shift_eval['overall']:.4f} "
                f"val={val_shift_eval['overall']:.4f} ({dt3+dt4:.1f}s)",
                flush=True,
            )
            print(
                "    per-dim source:",
                {k: round(v, 4) for k, v in src_eval["per_dim"].items()},
                flush=True,
            )
            print(
                "    per-dim val   :",
                {k: round(v, 4) for k, v in val_eval["per_dim"].items()},
                flush=True,
            )
            metrics.append(
                {
                    "step": step,
                    "source_action_mse": src_mse,
                    "val_action_mse": val_mse,
                    "style_gap": val_mse - src_mse,
                    "source_per_dim": src_eval["per_dim"],
                    "val_per_dim": val_eval["per_dim"],
                    "source_per_task": src_eval["per_task"],
                    "val_per_task": val_eval["per_task"],
                    "eval_shift": args.eval_shift,
                    "source_shift_mse": src_shift_eval["overall"],
                    "val_shift_mse": val_shift_eval["overall"],
                    "source_shift_per_dim": src_shift_eval["per_dim"],
                    "val_shift_per_dim": val_shift_eval["per_dim"],
                    "source_shift_per_task": src_shift_eval["per_task"],
                    "val_shift_per_task": val_shift_eval["per_task"],
                    "train_loss": float(loss),
                }
            )
            model.train()

    result = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "objective": OBJECTIVE,
        "images": image_contract(),
        "upstream": upstream,
        "initialization_sha256": initialization_sha256,
        "optimization": optimization_recipe,
        "support": {
            "train": train_ds.support(),
            "source": source_ds.support(),
            "val": val_ds.support(),
            "source_eval": source_ds.support(source_loader.dataset.indices),
            "val_eval": val_ds.support(val_loader.dataset.indices),
        },
        "merge": {k: merge[k] for k in ("copied", "random_init")},
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
    if args.run_test:
        test_ds = build_dataset(tasks, (41, 50), args.horizon, 224, True)
        test_loader = balanced_loader(
            test_ds, args.eval_batch_size, args.per_task_eval, 123
        )
        test_eval, dt = evaluate(
            model,
            test_loader,
            norm,
            device,
            len(test_loader),
            args.euler_steps,
            ema=ema,
        )
        result["n_test"] = len(test_ds)
        result["test_action_mse"] = test_eval["overall"]
        result["test_per_dim"] = test_eval["per_dim"]
        result["test_per_task"] = test_eval["per_task"]
        print(f"TEST action_mse={test_eval['overall']:.4f} ({dt:.1f}s)", flush=True)
        print(
            "  per-dim test:",
            {k: round(v, 4) for k, v in test_eval["per_dim"].items()},
            flush=True,
        )

    save_checkpoint(out / "model.pt", step)
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("DONE ->", out / "result.json", "| model.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
