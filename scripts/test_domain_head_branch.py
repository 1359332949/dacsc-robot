#!/usr/bin/env python3
"""Smoke test: frozen right-agentview DINOv2 domain branch."""
from __future__ import annotations
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from test_gating import build_policy, dummy_obs  # noqa: E402


def main():
    p, cfg = build_policy(
        enable_gating=False, use_resampler=True, flow_matching=True,
        resampler_num_queries=4, resampler_dim=64, num_inference_steps=2,
        domain_head_checkpoint="/hy-tmp/runs/domain_head_warmstart.pt",
        domain_head_feature="patch_mean", domain_head_log_every=1)
    p.train()
    obs = dummy_obs(cfg.shape_meta, B=4, To=2, seed=7)
    batch = {
        "obs": obs,
        "action": torch.randn(4, int(cfg.horizon), 12),
        "domain_id": torch.tensor([11, 16, 21, 26]),
        "layout_id": torch.tensor([11, 16, 21, 26]),
    }
    loss = p.compute_loss(batch)
    loss.backward()
    rep = p.domain_head_report()
    assert rep and "layout10_acc" in rep and "coarse3_acc" in rep, rep
    assert p.domain_head_branch is not None
    assert all(not q.requires_grad for q in p.domain_head_branch.dino.parameters())
    print("DOMAIN_HEAD_BRANCH_OK", rep)


if __name__ == "__main__":
    main()
