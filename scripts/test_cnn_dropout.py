#!/usr/bin/env python3
"""Integration tests for faithful CNN dropout on the official ResNet18 policy."""
from __future__ import annotations
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from test_gating import dummy_obs  # noqa: E402

CFG_DIR = ROOT / "repos" / "diffusion_policy" / "diffusion_policy" / "config"


def build(mode, **over):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="train_official_cnn_dropout",
            overrides=["task=robocasa/pretrain_atomic61"])
    OmegaConf.resolve(cfg)
    pol = dict(cfg.policy)
    pol["cnn_dropout_mode"] = mode
    pol.update(over)
    p = hydra.utils.instantiate(pol)
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    nz = LinearNormalizer()
    data = {k: torch.randn((8, 2) + tuple(v["shape"]))
            for k, v in cfg.shape_meta.obs.items()}
    data["action"] = torch.randn(
        8, int(cfg.horizon), int(cfg.shape_meta.action.shape[0]))
    nz.fit(data)
    p.set_normalizer(nz)
    return p, cfg


def batch_for(cfg, seed=7):
    obs = dummy_obs(cfg.shape_meta, B=4, To=2, seed=seed)
    return {"obs": obs,
            "action": torch.randn(4, int(cfg.horizon), int(cfg.shape_meta.action.shape[0])),
            "domain_id": torch.tensor([11, 11, 16, 16])}


def main():
    torch.manual_seed(0)

    # PLACE: random layer, progressive ratio.
    pp, cfg = build("place", cnn_layer_selection="random",
                    cnn_progressive=True, cnn_velocity=5.0,
                    cnn_drop_ratio=0.33, cnn_diagnostics_every=1)
    pp.train()
    b = batch_for(cfg)
    for _ in range(2):
        loss = pp.compute_loss(b)
        loss.backward()
        assert torch.isfinite(loss), loss
        pp.zero_grad(set_to_none=True)
    metrics = pp.cnn_report()["metrics"]
    assert metrics and any("keep_frac" in k for k in metrics), metrics
    print("PLACE_OK", metrics)

    # DomainDrop: domain head + GRL auxiliary loss.
    pd, cfg = build("domaindrop", cnn_layer_selection="fixed",
                    cnn_layer_indices=[7], cnn_num_domains=10,
                    cnn_domain_group_size=5, cnn_drop_ratio=0.33,
                    cnn_diagnostics_every=1)
    pd.train()
    loss = pd.compute_loss(b)
    loss.backward()
    assert torch.isfinite(loss), loss
    heads = [m.domain_heads[7][-1] for m in pd.cnn_controllers]
    assert all(h.weight.grad is not None and h.weight.grad.abs().sum() > 0
               for h in heads)
    print("DOMAINDROP_OK", pd.cnn_report()["metrics"])

    # DgCD: discrepancy-guided WRS on the same FiLM layer.
    pg, cfg = build("dgcd", cnn_layer_selection="fixed",
                    cnn_layer_indices=[7], cnn_drop_ratio=0.33,
                    cnn_diagnostics_every=1)
    pg.train()
    loss = pg.compute_loss(b)
    loss.backward()
    assert torch.isfinite(loss), loss
    print("DGCD_OK", pg.cnn_report()["metrics"])
    print("cnn dropout tests: PASS")


if __name__ == "__main__":
    main()
