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
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from dacsc.corp_data import (  # noqa: E402
    CorpRoboCasaDataset, LinearNormalizer, collate_corp,
)
from dacsc.corp_robocasa import (  # noqa: E402
    ROBOCASA_CORP_CONFIG, load_corp_robocasa,
)

DATA_ROOT = Path("/hy-tmp/datasets/robocasa365/v1.0/pretrain/atomic")
CKPT = Path("/hy-tmp/models/corp/CoRP/checkpoints/corp_libero.pt")


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
    }


@torch.no_grad()
def evaluate(model, loader, norm, device, max_batches, euler_steps):
    model.eval()
    total, n = 0.0, 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        b = move_batch(batch, norm, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.sample_action(b["images"], b["state"], b["instruction"],
                                       num_steps=euler_steps)
        pred = norm.unnormalize_action(pred.float())
        target = b["action"].float()
        total += F.mse_loss(pred, target, reduction="sum").item()
        n += target.numel()
    return total / max(n, 1), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=51)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--max-val-batches", type=int, default=16)
    ap.add_argument("--max-test-batches", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--euler-steps", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default="/hy-tmp/runs/corp_robocasa_p2")
    ap.add_argument("--run-test", action="store_true")
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

    model, merge = load_corp_robocasa(CKPT, config=ROBOCASA_CORP_CONFIG, device=device)
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
    val_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_corp, pin_memory=True,
    )
    source_loader = DataLoader(
        source_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_corp, pin_memory=True,
    )

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
                b["images"], b["state"], b["instruction"], b["action"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step {step:4d} loss {float(loss):.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} elapsed {time.time()-t_start:.1f}s",
                  flush=True)
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            src_mse, dt1 = evaluate(model, source_loader, norm, device,
                                    args.max_val_batches, args.euler_steps)
            val_mse, dt2 = evaluate(model, val_loader, norm, device,
                                    args.max_val_batches, args.euler_steps)
            print(f"  step {step}: source_mse={src_mse:.4f} val_mse={val_mse:.4f} "
                  f"gap={val_mse-src_mse:+.4f} ({dt1+dt2:.1f}s)", flush=True)
            metrics.append({"step": step, "source_action_mse": src_mse,
                            "val_action_mse": val_mse,
                            "style_gap": val_mse - src_mse,
                            "train_loss": float(loss)})
            model.train()

    result = {"args": vars(args), "merge": {k: merge[k] for k in ("copied", "random_init")},
              "n_train": len(train_ds), "n_val": len(val_ds), "metrics": metrics}
    if args.run_test:
        test_ds = build_dataset(tasks, (41, 50), args.horizon, 224, True)
        test_loader = DataLoader(
            test_ds, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_corp, pin_memory=True)
        test_mse, dt = evaluate(model, test_loader, norm, device,
                                args.max_test_batches, args.euler_steps)
        result["n_test"] = len(test_ds)
        result["test_action_mse"] = test_mse
        print(f"TEST action_mse={test_mse:.4f} ({dt:.1f}s)", flush=True)

    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("DONE ->", out / "result.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
