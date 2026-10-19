#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在本机（Apple M4 / MPS）跑通**官方训练循环**的烟测。

为什么这样做：真实 RoboCasa365 数据拿不到，但训练管线能否跑通是**独立**的问题。
这里用合成数据集（外观真的依赖域标签）走一遍官方 workspace：
    实例化 → 取 normalizer → DataLoader → 前向 → 反向 → 优化器（含门控参数）
    → EMA → 采样检查 → 存 checkpoint

跑通 = 数据一到就能直接开训。
"""
from __future__ import annotations
import json, os, shutil, sys, time
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")

import numpy as np
import torch
from omegaconf import OmegaConf

import dacsc.compat  # noqa

ROOT = Path(__file__).resolve().parent.parent
DP_CONFIG = ROOT / "repos" / "diffusion_policy" / "diffusion_policy" / "config"
OUT = ROOT / ".cache" / "train_smoke"


def main():
    steps = int(os.environ.get("SMOKE_STEPS", "6"))
    bs = int(os.environ.get("SMOKE_BS", "4"))
    n_dom = 4

    print("=" * 92)
    print(f"① 组合配置（batch={bs}, steps={steps}, domains={n_dom}）")
    import hydra
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize_config_dir(config_dir=str(DP_CONFIG), version_base=None):
        cfg = hydra.compose(config_name="train_dinov2_bs192")
    OmegaConf.set_struct(cfg, False)

    # --- 换成合成数据集（真实数据不可达）---
    cfg.task.dataset._target_ = "dacsc.stub_dataset.SyntheticDomainDataset"
    cfg.task.dataset.n_samples = 64
    cfg.task.dataset.n_domains = n_dom
    cfg.task.dataset.horizon = int(cfg.horizon)
    cfg.task.dataset.n_obs_steps = int(cfg.n_obs_steps)
    cfg.task.dataset.domain_effect = 0.35          # 外观真的依赖域
    cfg.task.dataset.action_domain_dependence = 0.0  # 动作与域无关（我们的假设）
    # --- 缩小规模适配本机 ---
    cfg.dataloader.batch_size = bs
    cfg.dataloader.num_workers = 0                 # Mac 上 0 最稳
    cfg.val_dataloader.batch_size = 2
    cfg.val_dataloader.num_workers = 0
    cfg.training.num_epochs = 1
    cfg.training.max_train_steps = steps
    cfg.training.checkpoint_every = steps
    cfg.training.sample_every = steps
    cfg.training.val_every = 10**9                 # 官方验证已被注释，这里无所谓
    cfg.training.debug = False
    cfg.logging.mode = "offline"
    cfg.policy.enable_gating = True
    cfg.policy.num_domains = n_dom
    cfg.policy.gate_rho_warmup_steps = max(1, steps)   # 让抑制预算真的推进
    # ★ 仅为在本机 8 步内**观察到门控变化**（验证学习通路），故意放大：
    #     gate_init 0.9 避开 sigmoid 饱和 + 门控 lr 放大。
    #     生产配置保持 gate_init=0.999 / gate_learning_rate=1e-2。
    cfg.policy.gate_init = 0.9
    cfg.policy.gate_learning_rate = 0.5
    OmegaConf.resolve(cfg)

    print(f"   policy._target_   = {cfg.policy._target_}")
    print(f"   dataset._target_  = {cfg.task.dataset._target_}")
    print(f"   enable_gating     = {cfg.policy.enable_gating}, freeze_dino={cfg.policy.freeze_dino}")

    print("\n② 实例化 workspace（官方训练入口）")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    cls = hydra.utils.get_class(cfg._target_)
    t0 = time.time()
    ws = cls(cfg, output_dir=str(OUT))
    print(f"   ✓ {cls.__name__} 实例化完成（{time.time()-t0:.1f}s）")
    from accelerate import Accelerator
    acc = Accelerator()
    print(f"   设备（accelerate 自选）= {acc.device}")
    # 记录训练前的门控值
    g_before = ws.model.gates().detach().cpu().clone() if ws.model.enable_gating else None
    if g_before is not None:
        print(f"   门控初始 g 均值 {float(g_before.mean()):.6f}，"
              f"形状 {tuple(g_before.shape)}")

    print("\n③ 跑官方 run()（训练 + EMA + 采样 + 存 checkpoint）")
    t0 = time.time()
    ws.run()
    dt = time.time() - t0
    print(f"\n   ✓ run() 结束，耗时 {dt:.1f}s")

    print("\n④ 检查产物")
    rep = {"steps": steps, "batch_size": bs, "run_seconds": round(dt, 1),
           "device": str(acc.device),
           "steps_per_sec": round(steps / max(dt, 1e-6), 2)}
    ckpts = sorted(OUT.rglob("*.ckpt"))
    logs = sorted(OUT.rglob("logs.json.txt"))
    rep["n_checkpoints"] = len(ckpts)
    rep["checkpoint_names"] = [c.name for c in ckpts[:3]]
    print(f"   checkpoint: {len(ckpts)} 个  {[c.name for c in ckpts[:2]]}")
    if logs:
        lines = [json.loads(l) for l in logs[0].read_text().splitlines() if l.strip()]
        losses = [l["train_loss"] for l in lines if "train_loss" in l]
        rep["n_log_lines"] = len(lines)
        rep["loss_first"] = losses[0] if losses else None
        rep["loss_last"] = losses[-1] if losses else None
        rep["loss_decreased"] = bool(len(losses) >= 2 and losses[-1] < losses[0])
        print(f"   训练日志 {len(lines)} 行")
        if len(losses) >= 2:
            print(f"   loss 首/末 = {losses[0]:.4f} → {losses[-1]:.4f}  "
                  f"（{'✓ 下降' if losses[-1] < losses[0] else '⚠ 未下降（步数太少属正常）'}）")
        gate_lines = [l for l in lines if "gate" in json.dumps(l)]
        if gate_lines:
            print(f"   含门控信息的日志行: {len(gate_lines)}")

    print("\n⑤ 门控是否真的在学")
    if g_before is not None:
        g_after = ws.model.gates().detach().cpu().clone()
        d = (g_after - g_before)
        rep["gate_mean_before"] = float(g_before.mean())
        rep["gate_mean_after"] = float(g_after.mean())
        rep["gate_max_delta"] = float(d.abs().max())
        rep["gate_n_changed"] = int((d.abs() > 1e-6).sum())
        rep["suppression_after"] = float(1 - g_after.mean())
        u = ws.model.domain_stats.u_disc().cpu()
        rep["u_disc_mean"] = float(u.abs().mean())
        _summ = ws.model.domain_stats.summary()
        rep["u_disc_top5"] = _summ.get("u_disc_top5_layer_channel", [])
        rep["domain_score_mode"] = _summ.get("mode", "kl_gauss")
        print(f"   g 均值 {float(g_before.mean()):.6f} → {float(g_after.mean()):.6f}")
        print(f"   最大变化 {float(d.abs().max()):.2e}，"
              f"有 {int((d.abs()>1e-6).sum())}/{g_before.numel()} 个门控被更新")
        print(f"   → {'✓ 门控在训练中真的被更新了' if float(d.abs().max())>1e-5 else '⚠ 几乎没动，检查学习率/预算'}")
        print(f"   u^disc（模式 {rep['domain_score_mode']}）: 均值 {float(u.abs().mean()):.4f}")
        if rep["u_disc_top5"]:
            print(f"   最高分通道(层,通道,分数) = {rep['u_disc_top5'][:3]}")
        print(f"   → {'✓ 域敏感性已从合成数据中算出' if float(u.abs().mean())>0 else '✗ u^disc 全 0'}")
    if ckpts:
        ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        sd = ck.get("state_dicts", {}).get("model", {})
        gate_keys = [k for k in sd if "gate_bank" in k]
        rep["n_gate_tensors_in_ckpt"] = len(gate_keys)
        if gate_keys:
            g = torch.cat([sd[k].flatten() for k in gate_keys])
            gates = torch.sigmoid(g)
            rep["gate_mean_final"] = float(gates.mean())
            rep["gate_min_final"] = float(gates.min())
            rep["suppression_final"] = float(1 - gates.mean())
            print(f"   checkpoint 里有 {len(gate_keys)} 个门控张量")
            print(f"   g 均值 {float(gates.mean()):.5f}，最小 {float(gates.min()):.5f}，"
                  f"抑制比例 {float(1-gates.mean()):.5f}")
            print(f"   （初始 g≈0.999 → 现在 {'已偏离初始值，门控在学习' if abs(float(gates.mean())-0.999)>1e-4 else '几乎没动'}）")

    out_p = ROOT / "audit" / "train_smoke.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n报告：{out_p}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
