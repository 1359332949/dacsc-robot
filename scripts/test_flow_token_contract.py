#!/usr/bin/env python3
"""Contract test: resampler must emit K tokens, not a flattened vector."""
from __future__ import annotations
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from test_gating import build_policy, dummy_obs  # noqa: E402


def main():
    p, cfg = build_policy(
        enable_gating=True, gate_mode="hard", num_domains=4,
        hard_layer_indices=[0, 3, 6, 9], hard_drop_ratio=0.33,
        hard_consistency_weight=0.0, domain_id_min=0,
        hard_mask_mode="gumbel_ste", action_importance_impl="rsc",
        hard_clean_importance_every=1, flow_matching=True,
        use_resampler=True, resampler_num_queries=4, resampler_dim=64,
        num_inference_steps=2)
    p.train()
    obs = dummy_obs(cfg.shape_meta, B=4, To=int(cfg.n_obs_steps), seed=7)
    batch = {
        "obs": obs,
        "action": torch.randn(4, int(cfg.horizon), 12),
        "domain_id": torch.tensor([0, 1, 2, 3]),
        "layout_id": torch.tensor([11, 12, 13, 14]),
    }
    nobs = p.normalizer.normalize(obs)
    this_nobs = {k: v[:, :p.n_obs_steps].reshape(-1, *v.shape[2:])
                 for k, v in nobs.items()}
    feat = p.obs_encoder(this_nobs)
    V = len(p.obs_encoder.rgb_keys)
    assert feat.shape[1] == 1 + V * 4, feat.shape
    assert feat.shape[2] == 64, feat.shape
    assert p.tokens_per_obs == 1 + V * 4
    assert p.model_obs_tokens == p.n_obs_steps * p.tokens_per_obs
    loss = p.compute_loss(batch)
    loss.backward()
    assert any(q.grad is not None for q in p.obs_encoder.resampler.parameters())
    p.eval()
    with torch.no_grad():
        out = p.predict_action(obs)
    assert out["action"].shape == (4, p.n_action_steps, p.action_dim)
    assert out["action_pred"].shape == (4, p.horizon, p.action_dim)
    print("FLOW_TOKEN_CONTRACT_OK", {
        "tokens_per_obs": p.tokens_per_obs,
        "model_obs_tokens": p.model_obs_tokens,
        "obs_feature_dim": p.obs_feature_dim,
        "loss": float(loss.detach()),
    })


if __name__ == "__main__":
    main()
