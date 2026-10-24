"""CoRP-RoboCasa 无门控 baseline（P2）。

严格 split：train styles 11-30 / val 31-40 / test 41-50。
模型：官方 CoRP 架构 + corp_libero.pt 共享权重，接口层随机初始化。
训练：rectified-flow velocity loss；动作 identity 归一化（对齐 DP 基线）。
"""
from __future__ import annotations

import argparse
import json
import random
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
    IMAGENET_MEAN, IMAGENET_STD, CorpRoboCasaDataset, LinearNormalizer,
    apply_image_shift, collate_corp,
)
from dacsc.corp_robocasa import (  # noqa: E402
    ROBOCASA_CORP_CONFIG, load_corp_robocasa,
)

DATA_ROOT = Path("/hy-tmp/datasets/robocasa365/v1.0/pretrain/atomic")
CKPT = Path("/hy-tmp/models/corp/CoRP/checkpoints/corp_libero.pt")

# action 维度顺序（shape_meta 拼接顺序）。
ACTION_GROUPS: dict[str, slice] = {
    "eef_position": slice(0, 3),
    "eef_rotation": slice(3, 6),
    "gripper_close": slice(6, 7),
    "base_motion": slice(7, 11),
    "control_mode": slice(11, 12),
}


def all_task_paths(max_tasks: int = 0) -> list[str]:
    paths = []
    for task_dir in sorted(DATA_ROOT.iterdir()):
        if not task_dir.is_dir():
            continue
        cand = sorted(task_dir.glob("*/lerobot"))
        if cand:
            paths.append(str(cand[0]))
        if max_tasks and len(paths) >= max_tasks:
            break
    return paths


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataset(tasks, split, horizon, image_size, use_cache,
                  episode_residues=None):
    lo, hi = split
    return CorpRoboCasaDataset(
        tasks, style_min=lo, style_max=hi, horizon=horizon,
        image_size=image_size, use_cache=use_cache,
        episode_residues=episode_residues,
    )


def move_batch(batch, norm, device):
    return {
        "images": {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()},
        "state": norm.normalize_state(batch["state"].to(device)),
        "instruction": batch["instruction"],
        "action": batch["action"].to(device),
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
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_corp, pin_memory=True,
    )


@torch.no_grad()
def evaluate(model, loader, norm, device, max_batches, euler_steps,
             shift: str = "none"):
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.sample_action(images, b["state"], b["instruction"],
                                       num_steps=euler_steps,
                                       domain_ids=b["style_id"])
        pred = norm.unnormalize_action(pred.float())
        target = b["action"].float()
        se = (pred - target) ** 2
        total += se.sum().item()
        n += target.numel()
        for g, sl in ACTION_GROUPS.items():
            group_sum[g] += se[..., sl].sum().item()
            group_n[g] += target[..., sl].numel()
        for j, task in enumerate(batch["task"]):
            task_sum[task] = task_sum.get(task, 0.0) + se[j].sum().item()
            task_n[task] = task_n.get(task, 0) + se[j].numel()
    out = {
        "overall": total / max(n, 1),
        "per_dim": {g: group_sum[g] / max(group_n[g], 1) for g in ACTION_GROUPS},
        "per_task": {t: task_sum[t] / max(task_n[t], 1) for t in task_sum},
    }
    return out, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=51)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--max-val-batches", type=int, default=16)
    ap.add_argument("--max-test-batches", type=int, default=16)
    ap.add_argument("--per-task-eval", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--euler-steps", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default="/hy-tmp/runs/corp_robocasa_p2")
    ap.add_argument("--run-test", action="store_true")
    ap.add_argument("--token-style-mode", default="none",
                    choices=["none", "random_stats", "cross_style_stats",
                             "cross_style_stats_action_aware", "normalize",
                             "normalize_action_aware"])
    ap.add_argument("--token-style-alpha", type=float, default=0.5)
    ap.add_argument("--token-style-action-weight", type=float, default=1.0)
    ap.add_argument("--token-style-protect-ratio", type=float, default=0.0)
    ap.add_argument("--token-style-stats", default="meanstd",
                    choices=["meanstd", "mean", "std"])
    ap.add_argument("--eval-shift", default="color_jitter",
                    choices=["none", "grayscale", "color_jitter",
                             "low_contrast", "hue_shift"])
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "P2 baseline 必须用 GPU"

    tasks = all_task_paths(args.max_tasks)
    print(f"tasks={len(tasks)}", flush=True)
    t0 = time.time()
    train_ds = build_dataset(tasks, (11, 30), args.horizon, 224, True,
                             episode_residues=set(range(9)))       # 90% source train
    source_ds = build_dataset(tasks, (11, 30), args.horizon, 224, True,
                              episode_residues={9})                # 10% source holdout
    val_ds = build_dataset(tasks, (31, 40), args.horizon, 224, True)
    print(f"datasets built in {time.time()-t0:.1f}s: train={len(train_ds)} "
          f"source_val={len(source_ds)} val={len(val_ds)}", flush=True)

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
    model, merge = load_corp_robocasa(
        CKPT, config=ROBOCASA_CORP_CONFIG, device=device, token_style=token_style)
    if token_style is not None:
        assert model._style_handle is not None, "token_style hook not registered"
    model.train()
    print(f"model loaded: copied={merge['copied']} random={merge['random_init']}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.steps // 10)))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_corp,
        pin_memory=True, drop_last=True,
    )
    # 评测用 per-task 均衡采样，保证 61 个 task 都被覆盖。
    val_loader = balanced_loader(val_ds, args.eval_batch_size,
                                 args.per_task_eval, args.seed + 1)
    source_loader = balanced_loader(source_ds, args.eval_batch_size,
                                    args.per_task_eval, args.seed + 2)

    metrics = []
    it = iter(train_loader)
    step = 0
    t_start = time.time()
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        b = move_batch(batch, norm, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.flow_matching_loss(
                b["images"], b["state"], b["instruction"], b["action"],
                domain_ids=b["style_id"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if getattr(model, "token_style", None) is not None:
            model.update_action_importance()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step {step:4d} loss {float(loss):.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} elapsed {time.time()-t_start:.1f}s",
                  flush=True)
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            src_eval, dt1 = evaluate(model, source_loader, norm, device,
                                     len(source_loader), args.euler_steps)
            val_eval, dt2 = evaluate(model, val_loader, norm, device,
                                     len(val_loader), args.euler_steps)
            src_mse, val_mse = src_eval["overall"], val_eval["overall"]
            print(f"  step {step}: source_mse={src_mse:.4f} val_mse={val_mse:.4f} "
                  f"gap={val_mse-src_mse:+.4f} ({dt1+dt2:.1f}s)", flush=True)
            src_shift_eval, dt3 = evaluate(
                model, source_loader, norm, device, len(source_loader),
                args.euler_steps, shift=args.eval_shift)
            val_shift_eval, dt4 = evaluate(
                model, val_loader, norm, device, len(val_loader),
                args.euler_steps, shift=args.eval_shift)
            print(f"    shifted({args.eval_shift}): "
                  f"source={src_shift_eval['overall']:.4f} "
                  f"val={val_shift_eval['overall']:.4f} ({dt3+dt4:.1f}s)", flush=True)
            print("    per-dim source:", {k: round(v, 4) for k, v in src_eval["per_dim"].items()},
                  flush=True)
            print("    per-dim val   :", {k: round(v, 4) for k, v in val_eval["per_dim"].items()},
                  flush=True)
            metrics.append({"step": step, "source_action_mse": src_mse,
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
                            "train_loss": float(loss)})
            model.train()

    result = {"args": vars(args), "merge": {k: merge[k] for k in ("copied", "random_init")},
              "n_train": len(train_ds), "n_val": len(val_ds), "metrics": metrics}
    if args.run_test:
        test_ds = build_dataset(tasks, (41, 50), args.horizon, 224, True)
        test_loader = balanced_loader(test_ds, args.eval_batch_size,
                                      args.per_task_eval, args.seed + 3)
        test_eval, dt = evaluate(model, test_loader, norm, device,
                                 len(test_loader), args.euler_steps)
        result["n_test"] = len(test_ds)
        result["test_action_mse"] = test_eval["overall"]
        result["test_per_dim"] = test_eval["per_dim"]
        result["test_per_task"] = test_eval["per_task"]
        print(f"TEST action_mse={test_eval['overall']:.4f} ({dt:.1f}s)", flush=True)
        print("  per-dim test:", {k: round(v, 4) for k, v in test_eval["per_dim"].items()},
              flush=True)

    torch.save({
        "format": "corp_robocasa_adapted_v1",
        "config": ROBOCASA_CORP_CONFIG,
        "model": model.state_dict(),
        "normalizer": {
            "state_mean": norm.state_mean.cpu(),
            "state_std": norm.state_std.cpu(),
        },
        "token_style": token_style_cfg,
    }, out / "model.pt")
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("DONE ->", out / "result.json", "| model.pt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
