#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 hard gate 单元测试（CPU/MPS）。"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dacsc.hard_gate import (HardMaskController, topk_ste_mask,  # noqa: E402
                             gumbel_ste_mask, wrs_mask, mask_iou_by_domain,
                             score_separation_by_domain)


def _ref_iou(mask, did, drop=False):
    """Old O(N^2) implementation, kept only as a test oracle."""
    keep = (mask <= 0) if drop else (mask > 0)
    same, diff = [], []
    n = keep.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            inter = (keep[i] & keep[j]).sum().item()
            union = (keep[i] | keep[j]).sum().item()
            if union == 0:
                continue
            v = inter / union
            (same if int(did[i]) == int(did[j]) else diff).append(v)
    a = sum(same) / len(same) if same else 0.0
    b = sum(diff) / len(diff) if diff else 0.0
    return a, b, a - b


def _ref_score_cos(scores, did):
    z = torch.nn.functional.normalize(scores.float(), dim=1)
    same, diff = [], []
    n = z.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            v = float((z[i] * z[j]).sum())
            (same if int(did[i]) == int(did[j]) else diff).append(v)
    a = sum(same) / len(same) if same else 0.0
    b = sum(diff) / len(diff) if diff else 0.0
    return a, b, a - b
from test_gating import build_policy, dummy_obs  # noqa: E402


def main():
    torch.manual_seed(0)
    # 1) top-k mask: 33% 丢弃、每样本不同、重归一化
    scores = torch.randn(5, 384)
    mask = topk_ste_mask(scores, drop_ratio=0.33, temperature=0.1)
    keep = (mask > 0).sum(dim=1)
    assert mask.shape == scores.shape
    expected = 384 - round(0.33 * 384)
    assert int(keep[0]) == expected, (int(keep[0]), expected)
    assert torch.allclose(mask.sum(dim=1), torch.full((5,), 384.0), atol=1e-4)
    assert not torch.equal((mask > 0)[0], (mask > 0)[1])
    print("① top-k STE mask: ✓", "keep", keep.tolist())
    gscores = torch.randn(4, 384)
    gscores[1] = gscores[0]
    gscores[3] = gscores[2]
    gids = torch.tensor([0, 0, 1, 1])
    gm = gumbel_ste_mask(gscores, drop_ratio=0.33, temperature=0.1,
                         group_ids=gids, domain_shared=True)
    assert torch.equal((gm[0] > 0), (gm[1] > 0))
    assert torch.equal((gm[2] > 0), (gm[3] > 0))
    assert not torch.equal((gm[0] > 0), (gm[2] > 0))
    print("①b domain-shared Gumbel: ✓")
    wm = wrs_mask(torch.randn(5, 384), drop_ratio=0.33)
    expected_keep = 384 - int(0.33 * 384)
    assert wm.shape == (5, 384)
    assert torch.equal((wm > 0).sum(dim=1), torch.full((5,), expected_keep))
    assert torch.allclose(wm.sum(dim=1), torch.full((5,), 384.0), atol=1e-4)
    wg = wrs_mask(torch.randn(5, 384), drop_ratio=0.33, normalize="global")
    assert torch.allclose(wg.sum(), torch.tensor(5 * 384.0), atol=1e-4)
    print("①c official-style WRS: ✓", "per-sample keep", expected_keep)

    # 1d) vectorized diagnostics must match the old pair loop exactly enough.
    torch.manual_seed(3)
    m = topk_ste_mask(torch.randn(12, 32), drop_ratio=0.33, temperature=0.1)
    did = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3])
    for drop in (False, True):
        got = mask_iou_by_domain(m, did, drop=drop)
        ref = _ref_iou(m, did, drop=drop)
        assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-5, (got, ref)
    sc = torch.randn(12, 32)
    got = score_separation_by_domain(sc, did)
    ref = _ref_score_cos(sc, did)
    assert max(abs(a - b) for a, b in zip(got, ref)) < 1e-5, (got, ref)
    # Empty-union pairs are ignored rather than counted as zero IoU.
    assert mask_iou_by_domain(torch.zeros(4, 8), torch.tensor([0, 0, 1, 1])) == (0.0, 0.0, 0.0)
    print("①d vectorized IoU/score diagnostics match pair-loop oracle: ✓")

    # 2) 训练/推理开关
    ctl = HardMaskController([0, 3, 6, 9], train_only=True)
    layer = ctl.select_layer()
    ctl.set_mask(mask[:2])
    x = torch.ones(2, 4, 384)
    y = ctl.apply(x, layer, training=True)
    y_eval = ctl.apply(x, layer, training=False)
    assert not torch.equal(x, y)
    assert torch.equal(x, y_eval)
    print("② train-time dropout / eval off: ✓", "layer", layer)

    # 3) policy hard 前向 + 反向 + 域头梯度
    p, cfg = build_policy(enable_gating=True, gate_mode="hard", num_domains=4,
                          hard_drop_ratio=0.33, hard_layer_indices=[0, 3, 6, 9],
                          hard_consistency_weight=0.0, domain_id_min=0,
                          hard_mask_mode="gumbel_ste", action_importance_impl="rsc",
                          hard_clean_importance_every=1)
    p.train()
    obs = dummy_obs(cfg.shape_meta, B=4, To=2, seed=7)
    batch = {"obs": obs, "action": torch.randn(4, int(cfg.horizon), 12),
             "domain_id": torch.tensor([0, 1, 2, 3])}
    p._collect.clear()
    loss = p.compute_loss(batch)
    loss.backward()
    rep = p.gate_report()
    assert rep["mode"] == "hard"
    assert rep["metrics"]["mask_keep_fraction"] > 0.6
    assert rep["metrics"]["suppression"] > 0.3
    assert p.domain_head.weight.grad is not None
    assert p.domain_head.weight.grad.abs().sum() > 0
    sd = p.state_dict()
    assert "domain_head.weight" in sd
    assert any(k.startswith("_dacsc_action_importance") for k in sd)
    assert rep["metrics"]["mask_mode"] == "gumbel_ste"
    assert "mask_iou_sample_gap" in rep["metrics"]
    assert "score_cos_gap" in rep["metrics"]
    print("③ hard policy loss/backward: ✓", "loss", round(float(loss.detach()), 4),
          "metrics", json.dumps(rep["metrics"], default=str)[:300])

    # 4) clean Taylor + RSC 通道分位数：v 必须 ready 且非塌缩
    p._collect.clear()
    p.compute_loss(batch).backward()
    ai = p.action_importance.summary()
    assert ai["ready"] and ai["v_std"] > 0.1 and ai["raw_nonzero"] > 0, ai
    print("④ action importance:", ai)
    # 5) classifier domain head must still train when action importance is off.
    pc, _ = build_policy(enable_gating=True, gate_mode="hard", num_domains=4,
                         hard_drop_ratio=0.33, hard_layer_indices=[0, 3, 6, 9],
                         hard_consistency_weight=0.0, domain_id_min=0,
                         hard_mask_mode="gumbel_ste", action_importance_impl="rsc",
                         use_action_importance=False, hard_domain_mode="classifier")
    pc.train()
    pc._collect.clear()
    loss_c = pc.compute_loss(batch)
    loss_c.backward()
    assert pc.domain_head.weight.grad is not None
    assert pc.domain_head.weight.grad.abs().sum() > 0
    print("⑤ classifier trains without action importance: ✓")

    # 6) O(N^2) diagnostics run only every N steps; other steps carry forward.
    pd, _ = build_policy(enable_gating=True, gate_mode="hard", num_domains=4,
                         hard_drop_ratio=0.33, hard_layer_indices=[0, 3, 6, 9],
                         hard_consistency_weight=0.0, domain_id_min=0,
                         hard_mask_mode="gumbel_ste", action_importance_impl="rsc",
                         hard_clean_importance_every=1,
                         gate_diagnostics_every=2)
    pd.train()
    for step in range(3):
        pd._collect.clear()
        pd.compute_loss(batch).backward()
        info = pd._current_mask_info
        assert info["diagnostics_done"] == (step % 2 == 0), (step, info["diagnostics_done"])
        expected_step = step if step % 2 == 0 else 0
        assert info["diag"]["step"] == expected_step, (step, info["diag"]["step"])
    print("⑥ diagnostics cadence + carry-forward: ✓")
    print("hard gate tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
