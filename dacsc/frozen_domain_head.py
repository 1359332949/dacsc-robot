"""Frozen DINOv2 right-agentview domain branch + warm-started heads.

The branch is deliberately independent of the policy backbone:
  right agentview -> center crop 224 -> frozen DINOv2 ViT-S/14
  -> patch mean -> warm-start standardization -> layout10/coarse3 logits
No language, FiLM, task condition, random crop, or main-policy gradients.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dacsc.dinov2_encoder import load_dinov2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def center_crop_224(x: torch.Tensor) -> torch.Tensor:
    c = 224
    h, w = x.shape[-2:]
    if h < c or w < c:
        return F.interpolate(x, size=(c, c), mode="bilinear", align_corners=False)
    i, j = (h - c) // 2, (w - c) // 2
    return x[..., i:i + c, j:j + c]


class FrozenDINOv2DomainHead(nn.Module):
    def __init__(self, checkpoint: str | Path, feature: str = "patch_mean"):
        super().__init__()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if feature not in ("patch_mean", "cls"):
            raise ValueError(f"unknown domain_head_feature={feature}")
        self.feature = feature
        self.register_buffer("feature_mean", ckpt["mean"].float())
        self.register_buffer("feature_std", ckpt["std"].float())
        for name in ("layout10", "coarse3"):
            head = ckpt[name]
            self.register_buffer(f"{name}_weight", head["weight"].float())
            self.register_buffer(f"{name}_bias", head["bias"].float())
            self.register_buffer(f"{name}_mean", head.get("mean", ckpt["mean"]).float())
            self.register_buffer(f"{name}_std", head.get("std", ckpt["std"]).float())
            self.register_buffer(f"{name}_temperature",
                                 torch.tensor(float(head.get("temperature", 1.0))))
        self.register_buffer("img_mean", IMAGENET_MEAN, persistent=False)
        self.register_buffer("img_std", IMAGENET_STD, persistent=False)
        self.dino = load_dinov2("dinov2_vits14", pretrained=True)
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.dino.eval()

    def train(self, mode: bool = True):
        # The domain branch must stay in eval/no-augmentation mode even when the
        # policy is training.
        super().train(False)
        self.dino.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """images: (N,3,H,W) raw [0,1] -> logits and feature norm."""
        x = center_crop_224(images.float())
        x = (x - self.img_mean.to(x)) / self.img_std.to(x)
        out = self.dino.forward_features(x)
        feat = (out["x_norm_patchtokens"].mean(dim=1)
                if self.feature == "patch_mean" else out["x_norm_clstoken"])
        f_layout = ((feat - self.layout10_mean.to(feat))
                    / self.layout10_std.to(feat))
        f_coarse = ((feat - self.coarse3_mean.to(feat))
                    / self.coarse3_std.to(feat))
        layout = ((f_layout @ self.layout10_weight.to(feat).t()
                   + self.layout10_bias.to(feat))
                  / self.layout10_temperature.to(feat))
        coarse = ((f_coarse @ self.coarse3_weight.to(feat).t()
                   + self.coarse3_bias.to(feat))
                  / self.coarse3_temperature.to(feat))
        return {"layout10": layout, "coarse3": coarse,
                "feature_norm": f_coarse.norm(dim=1).mean()}


__all__ = ["FrozenDINOv2DomainHead", "center_crop_224"]
