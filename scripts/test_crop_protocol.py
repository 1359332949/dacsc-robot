#!/usr/bin/env python3
"""Verify that baseline center crop is identical in train and eval mode."""
from __future__ import annotations

import torch
from omegaconf import OmegaConf
from pathlib import Path

from dacsc.dinov2_encoder import _Crop224

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "repos/diffusion_policy/diffusion_policy/config"


def main():
    x = torch.arange(3 * 256 * 256, dtype=torch.float32).reshape(1, 3, 256, 256)
    crop = _Crop224(224, random_crop=False)
    crop.train()
    train_out = crop(x)
    crop.eval()
    eval_out = crop(x)
    assert torch.equal(train_out, eval_out), (
        "center-crop baseline must produce identical train/eval crops")
    assert train_out.shape == (1, 3, 224, 224)

    for name in ("train_dinov2_flow.yaml", "train_dinov2_bs192.yaml"):
        cfg = OmegaConf.load(CONFIG_DIR / name)
        assert cfg.policy.random_crop is False, (name, cfg.policy.random_crop)
        assert int(cfg.policy.crop_size) == 224
    print("CROP_PROTOCOL_OK", {"crop_size": 224, "random_crop": False})


if __name__ == "__main__":
    main()
