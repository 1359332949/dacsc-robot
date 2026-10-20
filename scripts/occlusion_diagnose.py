#!/usr/bin/env python3
"""Causal occlusion test for domain-sensitive vs action-important dimensions."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import dill
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import dacsc.compat  # noqa: E402


def build_policy_and_batch(ckpt: Path, batch_size: int, device: str):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg_dir = ROOT / "repos/diffusion_policy/diffusion_policy/config"
    with hydra.initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = hydra.compose(
            config_name="train_dinov2_flow",
            overrides=["task=robocasa/pretrain_atomic61"])
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    policy = hydra.utils.instantiate(cfg.policy)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)

    payload = torch.load(ckpt.open("rb"), map_location="cpu", pickle_module=dill)
    state = payload["state_dicts"]["model"]
    missing, unexpected = policy.load_state_dict(state, strict=False)
    # checkpoint state_dicts 是从 CPU 复制的，加载后必须重新搬到 GPU。
    policy.to(device)
    print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}")
    sampler = dataset.get_dataset_sampler()
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    def move(obj):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: move(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(move(v) for v in obj)
        return obj
    batch = move(batch)
    return policy, batch


def scores_for_batch(policy, batch, layer: int | None):
    policy.train()
    policy._stats_update_enabled = False
    policy._current_domain_ids = batch.get("domain_id", None)
    policy._current_layout_ids = batch.get("layout_id", None)
    policy._current_mask_info = {}
    policy._collect.clear()
    layer = policy.hard_controller.select_layer() if layer is None else int(layer)
    policy.hard_controller.layer = int(layer)
    policy.hard_controller.set_mask(None)
    policy.hard_controller.build_enabled = True
    with torch.no_grad():
        policy._base_loss(batch)
    info = policy._current_mask_info
    if not info:
        raise RuntimeError("mask builder 没有返回 score info")
    return int(info["layer"]), info["u"].detach(), info["v"].detach(), info


def make_mask(scores: torch.Tensor, drop_ratio: float, mode: str,
              generator: torch.Generator):
    n, c = scores.shape
    k = max(1, min(c, int(round(drop_ratio * c))))
    if mode == "random":
        key = torch.rand(scores.shape, device=scores.device, generator=generator)
    else:
        key = scores
    idx = torch.topk(key, k, dim=1, largest=True).indices
    keep = torch.ones((n, c), device=scores.device, dtype=scores.dtype)
    keep.scatter_(1, idx, 0.0)
    return keep


def eval_loss(policy, batch, block, mask=None, seed=12345):
    handle = None
    if mask is not None:
        def hook(module, inputs, output):
            m = mask.to(device=output.device, dtype=output.dtype)
            if m.shape[0] != output.shape[0]:
                m = m[:output.shape[0]]
            return output * m[:, None, :]
        handle = block.register_forward_hook(hook)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            loss = policy._base_loss(batch)
        return float(loss.detach())
    finally:
        if handle is not None:
            handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/runs/occlusion.json"))
    ap.add_argument("--batch-size", type=int, default=30)
    ap.add_argument("--drop-ratio", type=float, default=0.33)
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    policy, batch = build_policy_and_batch(args.ckpt, args.batch_size, args.device)
    layer, u, v, info = scores_for_batch(policy, batch, args.layer)
    policy.disable_gating_hooks()
    policy.eval()
    block = policy.obs_encoder.dino.blocks[layer]
    gen = torch.Generator(device=u.device)
    gen.manual_seed(42)

    full = eval_loss(policy, batch, block)
    top_u = make_mask(u, args.drop_ratio, "top", gen)
    top_v = make_mask(v[None, :].expand_as(u), args.drop_ratio, "top", gen)
    random = make_mask(u, args.drop_ratio, "random", gen)
    losses = {
        "full": full,
        "top_u": eval_loss(policy, batch, block, top_u),
        "top_v": eval_loss(policy, batch, block, top_v),
        "random": eval_loss(policy, batch, block, random),
    }
    out = {
        "layer": layer,
        "drop_ratio": args.drop_ratio,
        "scores": {"u_std": float(u.std()), "v_std": float(v.std()),
                   "v_ready": bool(info.get("v_ready", False))},
        "loss": losses,
        "delta_vs_full": {k: losses[k] - full for k in losses if k != "full"},
        "delta_vs_random": {"top_u": losses["top_u"] - losses["random"],
                             "top_v": losses["top_v"] - losses["random"]},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"OUTPUT: {args.out}")


if __name__ == "__main__":
    main()
