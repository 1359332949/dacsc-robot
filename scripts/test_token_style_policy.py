#!/usr/bin/env python3
"""Integration smoke: token-style modulation inside the flow K-token policy."""
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
        token_style_mode="cross_style_stats_action_aware",
        token_style_alpha=0.5, token_style_action_weight=1.0,
        token_style_importance_every=1, token_style_diagnostics_every=1)
    p.train()
    obs = dummy_obs(cfg.shape_meta, B=4, To=int(cfg.n_obs_steps), seed=7)
    batch = {
        "obs": obs,
        "action": torch.randn(4, int(cfg.horizon), 12),
        "domain_id": torch.tensor([0, 1, 2, 3]),
        "layout_id": torch.tensor([11, 12, 13, 14]),
    }
    for step in range(2):
        loss = p.compute_loss(batch)
        loss.backward()
        assert torch.isfinite(loss), loss
        p.zero_grad(set_to_none=True)
    style = p.obs_encoder.token_style
    assert style is not None
    assert p.token_importance is not None and p.token_importance.ready()
    assert p.token_importance.normalized().shape == (4,)
    diff = float(style.last_stats_t["diff_domain_frac"])
    assert diff > 0.8, diff
    print("TOKEN_STYLE_POLICY_OK", {
        "loss": float(loss.detach()),
        "diff_domain_frac": diff,
        "importance": p.token_importance.summary(),
        "style": style.summary(),
    })

    # Patch-stage variant: style statistics are applied before the resampler.
    pp, _ = build_policy(
        enable_gating=False, use_resampler=True, flow_matching=True,
        resampler_num_queries=4, resampler_dim=64, num_inference_steps=2,
        token_style_mode="cross_style_stats", token_style_stage="patch",
        token_style_stats="mean", token_style_alpha=0.5,
        token_style_diagnostics_every=1)
    pp.train()
    loss_p = pp.compute_loss(batch)
    loss_p.backward()
    style_p = pp.obs_encoder.token_style
    diff_p = float(style_p.last_stats_t["diff_domain_frac"])
    assert diff_p > 0.8, diff_p
    print("TOKEN_STYLE_PATCH_OK", {
        "loss": float(loss_p.detach()),
        "diff_domain_frac": diff_p,
        "style": style_p.summary(),
    })

    # Canonical normalization must be trainable end-to-end.
    pn, _ = build_policy(
        enable_gating=False, use_resampler=True, flow_matching=True,
        resampler_num_queries=4, resampler_dim=64, num_inference_steps=2,
        token_style_mode="normalize", token_style_stage="resampler",
        token_style_diagnostics_every=1)
    pn.train()
    loss_n = pn.compute_loss(batch)
    loss_n.backward()
    gamma = pn.obs_encoder.token_style.norm_gamma
    beta = pn.obs_encoder.token_style.norm_beta
    assert gamma.grad is not None and beta.grad is not None
    assert gamma.grad.abs().sum() > 0
    print("TOKEN_STYLE_NORMALIZE_OK", {
        "loss": float(loss_n.detach()),
        "gamma_grad": float(gamma.grad.abs().sum()),
        "beta_grad": float(beta.grad.abs().sum()),
    })


if __name__ == "__main__":
    main()
