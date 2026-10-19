#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DINOv2 策略的 CPU 冒烟测试。

验证：
  1) hydra 能按官方配置路径实例化我们的策略（说明可插进官方训练/评测管线）
  2) DINOv2 编码器输出形状、参数分组正确
  3) compute_loss 能前向+反向（梯度能流到 DINOv2）
  4) predict_action 输出与官方格式一致（action [B, n_action_steps, 12]）

用法：
  ./.venv/bin/python scripts/test_dinov2_policy.py
  ./.venv/bin/python scripts/test_dinov2_policy.py --compare-official   # 同时跑官方基线对比
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch
from omegaconf import OmegaConf

import dacsc.compat  # noqa: F401  必须在导入策略类之前（补 diffusers 缺失符号）

ROOT = Path(__file__).resolve().parent.parent
DP_CONFIG = ROOT / "repos" / "diffusion_policy" / "diffusion_policy" / "config"


def build_cfg(task="robocasa/pretrain_human300", config_name="train_diffusion_transformer_bs192"):
    import hydra
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize_config_dir(config_dir=str(DP_CONFIG), version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=[f"task={task}"])
    OmegaConf.resolve(cfg)
    return cfg


def dummy_batch(shape_meta, B=2, To=2, horizon=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = {}
    for k, attr in shape_meta["obs"].items():
        shape = tuple(attr["shape"])
        obs[k] = torch.rand((B, To) + shape, generator=g)
    action = torch.randn(B, horizon, shape_meta["action"]["shape"][0], generator=g)
    return {"obs": obs, "action": action}


def fit_dummy_normalizer(shape_meta, horizon, B=8):
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    nz = LinearNormalizer()
    data = {}
    for k, attr in shape_meta["obs"].items():
        data[k] = torch.randn((B, 2) + tuple(attr["shape"]))
    data["action"] = torch.randn(B, horizon, shape_meta["action"]["shape"][0])
    nz.fit(data)
    return nz


def count(m): return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="robocasa/pretrain_human300")
    ap.add_argument("--compare-official", action="store_true")
    ap.add_argument("--no-backward", action="store_true")
    a = ap.parse_args()

    rep = {"task": a.task}
    print("=" * 88)
    print("① hydra 组合官方配置")
    cfg = build_cfg(a.task)
    print(f"   horizon={cfg.horizon} n_obs_steps={cfg.n_obs_steps} "
          f"n_action_steps={cfg.n_action_steps} batch_size={cfg.dataloader.batch_size}")
    print(f"   obs 键 {len(cfg.shape_meta.obs)} 个，action shape={cfg.shape_meta.action.shape}")

    print("\n② 把 policy._target_ 换成我们的类")
    OmegaConf.set_struct(cfg, False)      # 允许注入我们自己的配置键
    cfg.policy._target_ = "dacsc.policy.RoboCasaDINOv2Policy"
    cfg.policy.dino_backbone = "dinov2_vits14"
    cfg.policy.feature_dim = 256
    print(f"   policy._target_ = {cfg.policy._target_}")

    import hydra
    t0 = time.time()
    policy = hydra.utils.instantiate(cfg.policy)
    policy.eval()
    rep["instantiate_s"] = round(time.time() - t0, 1)
    print(f"   ✓ 实例化成功（{rep['instantiate_s']}s）")

    n_all, n_enc, n_dit = count(policy), count(policy.obs_encoder), count(policy.model)
    n_dino = count(policy.obs_encoder.dino)
    rep.update(params_total=n_all, params_encoder=n_enc, params_dino=n_dino, params_dit=n_dit)
    print(f"   总参数 {n_all/1e6:.2f} M  |  观测编码器 {n_enc/1e6:.2f} M"
          f"（其中 DINOv2 {n_dino/1e6:.2f} M）  |  DiT {n_dit/1e6:.2f} M")
    print(f"   obs_feature_dim = {policy.obs_feature_dim}")
    print(f"   get_optimizer 参数组数 = {len(policy.get_optimizer(1e-3,1e-6,1e-4,(0.9,0.95)).param_groups)}")

    print("\n③ compute_loss（前向 + 反向）")
    policy.set_normalizer(fit_dummy_normalizer(cfg.shape_meta, int(cfg.horizon)))
    print("   dummy normalizer 已设置（真实训练时由 dataset.get_normalizer() 提供）")
    batch = dummy_batch(cfg.shape_meta, B=2, To=int(cfg.n_obs_steps), horizon=int(cfg.horizon))
    policy.train()
    t0 = time.time()
    loss = policy.compute_loss(batch)
    rep["loss"] = float(loss)
    rep["fwd_s"] = round(time.time() - t0, 1)
    print(f"   loss = {float(loss):.4f}   前向耗时 {rep['fwd_s']}s")
    if not a.no_backward:
        t0 = time.time()
        loss.backward()
        rep["bwd_s"] = round(time.time() - t0, 1)
        g_dino = sum(p.grad.abs().sum().item() for p in policy.obs_encoder.dino.parameters()
                     if p.grad is not None)
        n_grad = sum(1 for p in policy.obs_encoder.dino.parameters() if p.grad is not None)
        rep["dino_grad_sum"] = g_dino
        rep["dino_params_with_grad"] = n_grad
        print(f"   反向耗时 {rep['bwd_s']}s")
        n_dino_params = sum(1 for _ in policy.obs_encoder.dino.parameters())
        print(f"   ★ 梯度能否流到 DINOv2：{n_grad}/{n_dino_params} 个参数有梯度"
              f"，梯度和 {g_dino:.3f}  → {'✓ 联合微调生效' if n_grad>0 else '✗ 梯度断了'}")

    print("\n④ predict_action（评测接口）")
    policy.eval()
    obs = {k: torch.randn((1, int(cfg.n_obs_steps)) + tuple(v["shape"]))
           for k, v in cfg.shape_meta.obs.items()}
    t0 = time.time()
    with torch.no_grad():
        out = policy.predict_action(obs)
    rep["predict_s"] = round(time.time() - t0, 1)
    rep["action_shape"] = list(out["action"].shape)
    rep["action_pred_shape"] = list(out["action_pred"].shape)
    print(f"   action      {rep['action_shape']}   （应为 [1, {cfg.n_action_steps}, 12]）")
    print(f"   action_pred {rep['action_pred_shape']}   （应为 [1, {cfg.horizon}, 12]）")
    print(f"   耗时 {rep['predict_s']}s")

    if a.compare_official:
        print("\n⑤ 对比官方基线（同样配置，只是不换编码器）")
        cfg2 = build_cfg(a.task)
        pol2 = hydra.utils.instantiate(cfg2.policy); pol2.eval()
        print(f"   官方：总参数 {count(pol2)/1e6:.2f} M | 观测编码器 {count(pol2.obs_encoder)/1e6:.2f} M"
              f" | obs_feature_dim {pol2.obs_feature_dim}")
        rep["official"] = {"params_total": count(pol2),
                           "params_encoder": count(pol2.obs_encoder),
                           "obs_feature_dim": pol2.obs_feature_dim}

    out_p = ROOT / "audit" / "dinov2_policy_smoke.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n报告：{out_p}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
