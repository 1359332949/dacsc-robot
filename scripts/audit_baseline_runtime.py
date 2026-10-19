#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 1 运行时审计：在 CPU 上真实实例化官方 Diffusion Policy 基线并做一次前向。

这解决的是"配置解析对了"之外的三个未知：
  1. 真实参数量（决定显存估算与训练可行性）
  2. 观测编码器 / DiT 的真实张量形状（决定门控插入点）
  3. 8 个 conv block 是否可被 hook（决定 DACSC 门控怎么注入）

依赖：torch, torchvision, diffusers, einops, hydra-core, omegaconf, numpy,
      transformers, h5py, termcolor, tqdm, matplotlib（+ 本仓库的
      repos/diffusion_policy 与 repos/robomimic 在 sys.path 上）

用法：
  python3 scripts/audit_baseline_runtime.py \
      --dp-repo repos/diffusion_policy --robomimic-repo repos/robomimic \
      --task robocasa/pretrain_human300 --out audit/baseline_runtime
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp-repo", type=Path, required=True)
    ap.add_argument("--robomimic-repo", type=Path, default=None)
    ap.add_argument("--config-name", default="train_diffusion_transformer_bs192")
    ap.add_argument("--task", default="robocasa/pretrain_human300")
    ap.add_argument("--out", type=Path, default=Path("audit/baseline_runtime"))
    ap.add_argument("--no-forward", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(args.dp_repo.resolve()))
    if args.robomimic_repo:
        sys.path.insert(0, str(args.robomimic_repo.resolve()))

    import hydra

    # --- 兼容 shim ---------------------------------------------------------
    # 官方 fork pin 的是 diffusers==0.11.1；新版 diffusers 把 lr_scheduler 里用到的
    # 几个符号挪走了。这里在内存里补回，避免修改官方仓库源码。
    # 审计报告里会记录本 shim，保证可追溯。
    import diffusers, diffusers.optimization as _dopt
    import typing, torch.optim as _topt
    shims = []
    for name, obj in (("Union", typing.Union), ("Optional", typing.Optional),
                      ("Optimizer", _topt.Optimizer)):
        if not hasattr(_dopt, name):
            setattr(_dopt, name, obj)
            shims.append(f"diffusers.optimization.{name}")

    # 官方 train.py 里注册的 eval resolver（配置中的 ${eval:'...'} 依赖它）
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    cfg_dir = str((args.dp_repo / "diffusion_policy" / "config").resolve())
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(config_name=args.config_name,
                            overrides=[f"task={args.task}"])
    OmegaConf.resolve(cfg)

    rep = {"config_name": args.config_name, "task": args.task}
    rep["compat_shims"] = shims
    rep["diffusers_version"] = getattr(diffusers, "__version__", "?")
    rep["resolved"] = {
        k: str(cfg.get(k)) for k in
        ("horizon", "n_obs_steps", "n_action_steps", "obs_as_cond")
    }
    rep["shape_meta"] = OmegaConf.to_container(cfg.shape_meta, resolve=True)

    print("[1] 实例化策略 ...")
    t0 = time.time()
    policy = hydra.utils.instantiate(cfg.policy)
    policy.eval()
    rep["instantiate_seconds"] = round(time.time() - t0, 1)

    n_all = sum(p.numel() for p in policy.parameters())
    n_enc = sum(p.numel() for p in policy.obs_encoder.parameters())
    rep["params_total"] = n_all
    rep["params_obs_encoder"] = n_enc
    rep["params_total_m"] = round(n_all / 1e6, 2)
    print(f"    总参数 {n_all/1e6:.2f} M（其中 obs_encoder {n_enc/1e6:.2f} M）")

    # 门控插入点：obs_encoder 下每个相机的 ResNet18ConvFiLM
    print("[2] 定位门控插入点 ...")
    blocks = []
    for name, mod in policy.obs_encoder.named_modules():
        cls = type(mod).__name__
        if "ConvFiLM" in cls:
            n_blk = len(getattr(mod, "_conv_blocks", []))
            ch = [b.out_channels if hasattr(b, "out_channels") else None
                  for b in getattr(mod, "_conv_blocks", [])]
            blocks.append({"module": name, "class": cls, "n_conv_blocks": n_blk,
                           "film_layers": len(getattr(mod, "_film_layers", []))})
    rep["gate_targets"] = blocks
    for b in blocks:
        print(f"    {b['class']}  @ {b['module']}  blocks={b['n_conv_blocks']} films={b['film_layers']}")

    # 逐 block 输出形状（用 hook 实测）
    if blocks:
        target = dict(policy.obs_encoder.named_modules())[blocks[0]["module"]]
        shapes = []
        handles = [blk.register_forward_hook(
            lambda m, i, o, _s=shapes: _s.append(tuple(o.shape)))
            for blk in target._conv_blocks]
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            x = target._base_block(dummy)
            for conv, film in zip(target._conv_blocks, target._film_layers):
                x = conv(x)
                x = film(x, torch.zeros(1, 768))
        for h in handles:
            h.remove()
        rep["conv_block_output_shapes_224"] = [list(s) for s in shapes]
        print(f"    224×224 输入下逐 block 输出：{[list(s) for s in shapes]}")

    if args.no_forward:
        out = args.out; out.mkdir(parents=True, exist_ok=True)
        (out / "runtime_report.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n报告：{out/'runtime_report.json'}")
        return

    # 前向：构造与 shape_meta 一致的假观测
    print("[3] 构造假观测并前向 ...")
    # predict_action 会走 normalizer，而 normalizer 平时由 dataset.get_normalizer() 设置。
    # 这里用与 shape_meta 一致的随机数据 fit 一个 dummy normalizer，以真实走通前向路径。
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    nz = LinearNormalizer()
    fit_data = {}
    for key, attr in rep["shape_meta"]["obs"].items():
        shape = tuple(attr["shape"])
        fit_data[key] = torch.randn(8, 2, *shape) if attr.get("type") == "rgb" else torch.randn(8, 2, *shape)
    fit_data["action"] = torch.randn(8, int(cfg.horizon), rep["shape_meta"]["action"]["shape"][0])
    nz.fit(fit_data)
    policy.set_normalizer(nz)
    rep["normalizer"] = "dummy (fit on random tensors; 仅用于冒烟测试)"

    obs = {}
    for key, attr in rep["shape_meta"]["obs"].items():
        shape = tuple(attr["shape"])
        obs[key] = torch.zeros((1, 2) + shape)      # (B, n_obs_steps, ...)
    t0 = time.time()
    with torch.no_grad():
        result = policy.predict_action(dict(obs))
    rep["forward_seconds"] = round(time.time() - t0, 2)
    rep["action_output"] = {k: list(v.shape) for k, v in result.items()}
    rep["sample_action_shape"] = list(result["action"].shape)
    print(f"    前向耗时 {rep['forward_seconds']}s，输出 {rep['action_output']}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "runtime_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n报告：{args.out/'runtime_report.json'}")


if __name__ == "__main__":
    main()
