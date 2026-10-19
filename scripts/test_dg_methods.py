#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""移植构件的正确性测试（CORAL / RSC / grad_reverse / DgCD 打分）。"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dacsc.dg_methods import coral_loss, rsc_grad_mask, grad_reverse, DgCDScorer

ROOT = Path(__file__).resolve().parent.parent


def main():
    rep = {}
    print("=" * 90)
    torch.manual_seed(0)

    print("① Deep CORAL（移植自 DomainBed）")
    a = torch.randn(32, 16)
    l_same = float(coral_loss([a.clone(), a.clone()]))
    b = torch.randn(32, 16) * 3 + 5
    l_diff = float(coral_loss([a, b]))
    rep.update(coral_same=l_same, coral_diff=l_diff)
    print(f"   两个相同域 → loss = {l_same:.3e}（应 ≈ 0）")
    print(f"   两个不同域 → loss = {l_diff:.4f}")
    ok1 = l_same < 1e-6 and l_diff > 0.1
    print(f"   → {'✓ 正确' if ok1 else '✗'}")

    print("\n② RSC 梯度分位数掩码（移植自 DomainBed）")
    f = torch.randn(16, 8, requires_grad=True)
    W = torch.randn(8, 4, requires_grad=True)
    logits = f @ W                       # ★ 必须由 features 算出（RSC 的前提）
    oh = torch.nn.functional.one_hot(torch.randint(0, 4, (16,)), 4).float()
    mask = rsc_grad_mask(f, logits, oh, drop_fraction=0.25)
    keep_ratio = float(mask.mean())
    rep["rsc_keep_ratio"] = keep_ratio
    print(f"   掩码形状 {tuple(mask.shape)}，保留比例 {keep_ratio:.3f}（drop=0.25 → 期望 ~0.75）")
    ok2 = mask.shape == f.shape and 0.6 < keep_ratio < 0.9
    print(f"   → {'✓ 正确' if ok2 else '✗'}")

    print("\n③ 梯度反转层")
    x = torch.randn(4, 3, requires_grad=True)
    y = grad_reverse(x, 2.0)
    y.sum().backward()
    rep["grad_reverse_value"] = float(x.grad.mean())
    print(f"   前向不变；反向梯度系数 = {float(x.grad.mean()):.1f}（应为 -2.0）")
    ok3 = abs(float(x.grad.mean()) + 2.0) < 1e-6
    print(f"   → {'✓ 正确' if ok3 else '✗'}")

    print("\n④ DgCD 打分（移植自 gyeomo/DgCD，CIKM'24）")
    sc = DgCDScorer(rho=1e-3)
    N, T, C = 32, 16, 24
    K = 6                       # 前 K 个通道"有域差异"，其余通道两域完全相同
    torch.manual_seed(1)
    base = torch.randn(N, T, C)
    d = torch.cat([torch.zeros(N // 2, dtype=torch.long),
                   torch.ones(N // 2, dtype=torch.long)])
    # (a) 单域 → 应为全 0
    s1 = sc.score(base, torch.zeros(N, dtype=torch.long)).detach()
    # (b) 受控：只有前 K 个通道在两域间有偏移
    f2 = base.clone()
    f2[N // 2:, :, :K] += 3.0
    s2 = sc.score(f2, d).detach()                     # (N, C)
    per_ch = s2.mean(dim=0)                           # (C,)
    hi = float(per_ch[:K].mean())                     # 真有域差异的通道
    lo = float(per_ch[K:].mean())                     # 无域差异的通道
    rep.update(dgcd_single_domain=float(s1.abs().mean()),
               dgcd_score_true_diff_channels=hi,
               dgcd_score_same_channels=lo, dgcd_ratio=hi / max(lo, 1e-9))
    print(f"   单域 → 分数 {float(s1.abs().mean()):.4f}（应为 0）")
    print(f"   前 {K} 个通道（**真有域差异**）的平均分 = {hi:.4f}")
    print(f"   其余 {C-K} 个通道（两域相同）的平均分 = {lo:.4f}")
    print(f"   比值 = {hi/max(lo,1e-9):.2f}×")
    ok4 = (float(s1.abs().mean()) == 0.0) and (hi > lo)
    print(f"   → {'✓ 正确：真正有域差异的通道得分更高' if ok4 else '✗ 未能区分'}")

    rep["all_ok"] = bool(ok1 and ok2 and ok3 and ok4)
    out = ROOT / "audit" / "dg_methods_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 90)
    print(f"总结：{'✓ 全部通过' if rep['all_ok'] else '✗ 有失败项'}   报告：{out}")
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
