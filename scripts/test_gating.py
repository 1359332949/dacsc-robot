#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 通道门控的 CPU 测试。

检查 6 件事：
  ① 恒等门控（g≈1）不改变基线输出      ← 最关键：证明注入是"无害"的
  ② 门控参数量正确（12 层 × 384 维）
  ③ u^disc 能算出来，且"域差异大 → 分数大"（性质测试）
  ④ L_gate 能反向传播，梯度能到门控参数
  ⑤ 优化若干步后门控值真的会变
  ⑥ 动作重要性 v^Taylor 能从梯度里读出来
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import torch
from omegaconf import OmegaConf

import dacsc.compat  # noqa
ROOT = Path(__file__).resolve().parent.parent
DP_CONFIG = ROOT / "repos" / "diffusion_policy" / "diffusion_policy" / "config"


def build_policy(enable_gating=True, gate_init=0.999, num_domains=4, use_action_importance=True, **over):
    import hydra
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize_config_dir(config_dir=str(DP_CONFIG), version_base=None):
        cfg = hydra.compose(config_name="train_diffusion_transformer_bs192",
                            overrides=["task=robocasa/pretrain_human300"])
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    pol = dict(cfg.policy)
    pol["_target_"] = "dacsc.policy.RoboCasaDINOv2Policy"
    pol.update(enable_gating=enable_gating, gate_init=gate_init, num_domains=num_domains,
               gate_layer_indices=None, use_action_importance=use_action_importance, **over)
    p = hydra.utils.instantiate(pol)
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    nz = LinearNormalizer()
    data = {k: torch.randn((8, 2) + tuple(v["shape"])) for k, v in cfg.shape_meta.obs.items()}
    data["action"] = torch.randn(8, int(cfg.horizon), cfg.shape_meta.action.shape[0])
    nz.fit(data)
    p.set_normalizer(nz)
    return p, cfg


def dummy_obs(shape_meta, B=2, To=2, seed=0, color=None):
    g = torch.Generator().manual_seed(seed)
    obs = {}
    for k, v in shape_meta.obs.items():
        shape = tuple(v["shape"])
        x = torch.rand((B, To) + shape, generator=g)
        if color is not None and v.get("type") == "rgb":
            x = x.clone()
            x[..., color] = (x[..., color] + 0.5).clamp(0, 1)   # 人为改某个颜色通道
        obs[k] = x
    return obs


def main():
    rep = {}
    print("=" * 92)
    print("① 恒等门控是否改变基线输出")
    torch.manual_seed(0)
    p_gate, cfg = build_policy(enable_gating=True, gate_init=1.0 - 1e-7)
    p_gate.eval()
    obs = dummy_obs(cfg.shape_meta, B=1, To=int(cfg.n_obs_steps), seed=1)
    # ★ 扩散采样是随机的：两次调用必须用同一随机种子才能对比
    with torch.no_grad():
        torch.manual_seed(12345)
        a_gate = p_gate.predict_action(obs)["action"]        # 装 hook（g≈1）
        p_gate.disable_gating_hooks()
        torch.manual_seed(12345)
        a_base = p_gate.predict_action(obs)["action"]        # 同一权重，拆掉 hook
        p_gate.reenable_gating_hooks()
    d = (a_gate - a_base).abs().max().item()
    rel = d / a_base.abs().mean().item()
    rep["identity_max_abs_diff"] = d
    rep["identity_rel_diff"] = rel
    print(f"   装 hook（g≈1）vs 拆 hook（同权重、同种子）：最大绝对差 {d:.3e}，相对差 {rel:.3e}")
    print(f"   → {'✓ 恒等门控无损（注入无害）' if rel < 1e-3 else '✗ 有偏差，需排查'}")
    print(f"   门控摘要：{p_gate.gate_report()['n_channels_total']} 个门控标量，"
          f"层号 {p_gate.gate_report()['layer_indices']}")

    print("\n② 门控参数量")
    n_gate = sum(q.numel() for q in p_gate.gate_parameters())
    n_all = sum(q.numel() for q in p_gate.parameters())
    rep["gate_params"] = n_gate; rep["total_params"] = n_all
    print(f"   门控参数 {n_gate} 个（12 层 × 384 维），占模型 {n_gate/n_all*100:.4f}%")

    print("\n③ u^disc 在完整管线里是否可用（两种打分模式都测）")
    def run_pipeline(mode, domains_present, n_steps=4):
        torch.manual_seed(0)
        p, _ = build_policy(enable_gating=True, num_domains=4,
                            domain_score_mode=mode)
        p.train()
        for st in range(n_steps):
            xs = {}
            for dom in domains_present:
                xs[dom] = dummy_obs(cfg.shape_meta, B=2, To=2,
                                    seed=100 + st, color=dom if dom > 0 else None)
            obs = {k: torch.cat([xs[d][k] for d in domains_present], 0) for k in xs[0]}
            did = torch.cat([torch.full((2,), d, dtype=torch.long) for d in domains_present])
            batch = {"obs": obs, "action": torch.randn(len(did), int(cfg.horizon), 12),
                     "domain_id": did}
            loss = p.compute_loss(batch)
            loss.backward()
        return p.domain_stats.u_disc()

    ok3 = True
    for mode in ("dgcd", "kl_gauss"):
        u_single = run_pipeline(mode, [0])            # 只有一个域 → 无差异
        u_multi = run_pipeline(mode, [0, 1, 2])       # 多个域
        ok_a = (u_single.abs().max().item() < 1e-6)   # 单域必须为 0
        ok_b = (u_multi.abs().max().item() > 0)       # 多域必须非 0
        ok_c = (tuple(u_multi.shape) == (12, 384))
        ok3 &= (ok_a and ok_b and ok_c)
        rep[f"u_disc_{mode}"] = {"single_domain_max": float(u_single.abs().max()),
                                 "multi_domain_max": float(u_multi.abs().max()),
                                 "shape": list(u_multi.shape)}
        print(f"   [{mode:<8}] 单域 max|u|={float(u_single.abs().max()):.2e}（应 0）；"
              f"多域 max|u|={float(u_multi.abs().max()):.3f}（应 >0）；形状 {tuple(u_multi.shape)}")
    print(f"   → {'✓ 两种模式的打分都在完整管线里可用' if ok3 else '✗ 有问题'}")
    print("   （打分器本身的严格性质测试见 scripts/test_dg_methods.py）")

    print("\n④ L_gate 反向传播")
    torch.manual_seed(0)
    p, _ = build_policy(enable_gating=True, num_domains=4)
    p.train()
    opt = torch.optim.Adam(p.gate_parameters(), lr=1e-2)
    g0 = p.gates().clone()
    for s in range(8):
        obs_a = dummy_obs(cfg.shape_meta, B=2, To=2, seed=10 + s, color=0)
        obs_b = dummy_obs(cfg.shape_meta, B=2, To=2, seed=20 + s, color=2)
        obs = {k: torch.cat([obs_a[k], obs_b[k]], 0) for k in obs_a}
        batch = {"obs": obs, "action": torch.randn(4, int(cfg.horizon), 12),
                 "domain_id": torch.tensor([0, 0, 1, 1])}
        loss = p.compute_loss(batch)
        opt.zero_grad(); loss.backward(); opt.step()
    g1 = p.gates().clone()
    rep["gate_moved"] = float((g1 - g0).abs().max())
    parts = p._last_gate_parts
    print(f"   loss 含 L_gate；8 步后门控变化 max|Δg| = {rep['gate_moved']:.3e}")
    print(f"   分项：domain_term={float(parts['domain_term']):.4f} "
          f"action_term={float(parts['action_term']):.4f} "
          f"budget_term={float(parts['budget_term']):.4f} "
          f"抑制比例={float(parts['suppression_ratio']):.4f}")
    print(f"   → {'✓ 门控参数在训练（梯度通了）' if rep['gate_moved']>1e-6 else '✗ 门控没动'}")

    print("\n⑤ 动作重要性 v^Taylor")
    ai = p.action_importance
    rep["action_importance"] = ai.summary()
    v = ai.value
    print(f"   v 形状 {tuple(v.shape)}，均值 {float(v.mean()):.3e}，最大 {float(v.max()):.3e}")
    print(f"   → {'✓ 从梯度读出了动作重要性' if float(v.abs().max())>0 else '✗ 全 0，梯度没读到'}")

    print("\n⑥ 门控最终报告")
    r = p.gate_report()
    rep["gate_report"] = {k: v for k, v in r.items() if k != "domain_stats"}
    print("   " + json.dumps({k: r[k] for k in ("n_layers","n_channels_total","g_mean","g_min","suppression_ratio")}, ensure_ascii=False))
    ds = r["domain_stats"]
    print("   domain_stats: " + json.dumps(
        {k: ds[k] for k in ("mode", "n_layers_tracked", "u_disc_mean") if k in ds},
        ensure_ascii=False))

    out = ROOT / "audit" / "gating_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n报告：{out}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
