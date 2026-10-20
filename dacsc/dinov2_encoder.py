"""DINOv2 观测编码器（替代官方 DP fork 里 robomimic 的 ResNet18ConvFiLM）。

设计依据：
  - CoRP 论文（arXiv 2610.08183）4.3：预训练 ViT-S/14 比其它编码器高 ≥14.9 分
  - CoRP 4.4：**必须联合微调**，冻结会掉 20 分（故 freeze_dino 默认 False）
  - 多视角共享权重（CoRP 的做法）
  - 输入必须是 patch_size=14 的整数倍 → 224 正好（256 会报错）

v1 范围：只做「多视角 → DINOv2 → 池化 → 融合 → 全局特征向量」，
       与官方 DP 的 conditioning 格式保持完全一致（每个观测步一个 token）。
       后续 v2 再加 learned-query resampler（K 个 token）。
"""
from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

DINO_EMBED_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}
DINO_HUB_DIR = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
DINO_CKPT_DIR = os.path.expanduser("~/.cache/torch/hub/checkpoints")

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dinov2(backbone: str = "dinov2_vits14", pretrained: bool = True):
    """优先从本地加载（github.com 被拦时 torch.hub 默认路径会失败）。

    本地优先的原因：本环境 github.com 不可达，但代码包可从 codeload 拉、
    权重可从 Meta CDN 拉（实测 4.9 MB/s）。参见 scripts/fetch_dinov2.py。
    """
    weight_path = os.path.join(DINO_CKPT_DIR, f"{backbone}_pretrain.pth")
    if os.path.isdir(DINO_HUB_DIR) and (not pretrained or os.path.isfile(weight_path)):
        m = torch.hub.load(DINO_HUB_DIR, backbone, source="local", pretrained=False)
        if pretrained:
            sd = torch.load(weight_path, map_location="cpu")
            missing, unexpected = m.load_state_dict(sd, strict=False)
            assert not missing and not unexpected, (
                f"DINOv2 权重不匹配: missing={len(missing)} unexpected={len(unexpected)}")
        return m
    # 回退：走 torch.hub 远程（需要能访问 github.com）
    return torch.hub.load("facebookresearch/dinov2", backbone, pretrained=pretrained)


class _Crop224(nn.Module):
    """把 256×256 变成 224×224（DINOv2 要求 14 的整数倍）。

    训练时随机裁剪（与官方 CropRandomizer 的作用一致），推理时中心裁剪。
    """

    def __init__(self, crop: int = 224, random_crop: bool = True):
        super().__init__()
        self.crop = crop
        self.random_crop = random_crop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.crop
        if x.shape[-2:] == (c, c):
            return x
        if x.shape[-1] < c or x.shape[-2] < c:          # 比目标小 → 上采样
            return F.interpolate(x, size=(c, c), mode="bilinear", align_corners=False)
        if self.training and self.random_crop:           # 训练：随机裁剪
            i = torch.randint(0, x.shape[-2] - c + 1, (1,)).item()
            j = torch.randint(0, x.shape[-1] - c + 1, (1,)).item()
            return x[..., i:i + c, j:j + c]
        i = (x.shape[-2] - c) // 2                       # 推理：中心裁剪
        j = (x.shape[-1] - c) // 2
        return x[..., i:i + c, j:j + c]


class DINOv2ObsEncoder(nn.Module):
    """多视角 DINOv2（共享权重）+ 本体状态 + 语言嵌入 → 全局特征向量。

    forward 输入 obs_dict：{key: (B, ...)}，其中 rgb 键为 (B, 3, H, W)，
    low_dim 键为 (B, D)。输出 (B, feature_dim)。
    """

    def __init__(self,
                 shape_meta: dict,
                 dino_backbone: str = "dinov2_vits14",
                 crop_size: int = 224,
                 view_feat_dim: int = 256,
                 low_dim_feat_dim: int = 128,
                 feature_dim: int = 256,
                 freeze_dino: bool = False,
                 random_crop: bool = True,
                 pool: str = "mean",
                 pretrained: bool = True,
                 unfreeze_last_n_blocks: int | None = None,
                 use_resampler: bool = False,
                 resampler_num_queries: int = 48,
                 resampler_dim: int = 64,
                 resampler_heads: int = 6,
                 resampler_dropout: float = 0.0,
                 token_style_mode: str = "none",
                 token_style_alpha: float = 0.5,
                 token_style_action_weight: float = 1.0):
        super().__init__()
        assert pool in ("mean", "meanmax"), f"v1 只实现 mean/meanmax，收到 {pool}"
        self.shape_meta = shape_meta
        self.pool = pool
        self.feature_dim = feature_dim
        self.crop_size = crop_size

        obs_meta = shape_meta["obs"]
        self.rgb_keys: List[str] = [k for k, v in obs_meta.items()
                                    if v.get("type", "low_dim") == "rgb"]
        self.low_dim_keys: List[str] = [k for k, v in obs_meta.items()
                                        if v.get("type", "low_dim") != "rgb"]
        assert self.rgb_keys, "shape_meta 里没有 rgb 观测"

        # ---- DINOv2（多视角共享一套权重）----
        self.dino = load_dinov2(dino_backbone, pretrained=pretrained)
        assert dino_backbone in DINO_EMBED_DIM, f"未知 backbone {dino_backbone}"
        d = DINO_EMBED_DIM[dino_backbone]
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad_(not freeze_dino)
        self.freeze_dino = freeze_dino
        self.dino_pretrained = pretrained
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        if (not freeze_dino) and unfreeze_last_n_blocks is not None:
            n = int(unfreeze_last_n_blocks)
            assert n > 0, "unfreeze_last_n_blocks 必须为正整数"
            for p in self.dino.parameters():
                p.requires_grad_(False)
            for blk in self.dino.blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            if hasattr(self.dino, "norm"):
                for p in self.dino.norm.parameters():
                    p.requires_grad_(True)

        self.use_resampler = bool(use_resampler)
        self.resampler_num_queries = int(resampler_num_queries)
        self.resampler_dim = int(resampler_dim)
        self.token_dim = self.resampler_dim if self.use_resampler else self.feature_dim
        self.resampler = None
        self.token_style = None
        self.current_domain_ids = None
        if self.use_resampler:
            from dacsc.resampler import LearnedQueryResampler
            self.resampler = LearnedQueryResampler(
                in_dim=d, out_dim=self.resampler_dim,
                num_queries=self.resampler_num_queries,
                n_heads=int(resampler_heads), dropout=float(resampler_dropout),
                lang_dim=int(obs_meta["lang_emb"]["shape"][0])
                if "lang_emb" in obs_meta else None)
            if token_style_mode != "none":
                from dacsc.token_style import TokenStyleModulator
                self.token_style = TokenStyleModulator(
                    mode=token_style_mode, alpha=token_style_alpha,
                    action_weight=token_style_action_weight)

        self.crop = _Crop224(crop_size, random_crop)
        self.register_buffer("img_mean", _IMAGENET_MEAN.clone(), persistent=False)
        self.register_buffer("img_std", _IMAGENET_STD.clone(), persistent=False)

        n_views = len(self.rgb_keys)
        self.n_views = n_views
        per_view_in = d * (2 if pool == "meanmax" else 1)
        self.view_proj = nn.Sequential(
            nn.Linear(per_view_in, view_feat_dim), nn.ReLU(),
            nn.Linear(view_feat_dim, view_feat_dim)) if not self.use_resampler else None

        low_dim_total = sum(int(obs_meta[k]["shape"][0]) for k in self.low_dim_keys)
        self.low_dim_net = nn.Sequential(
            nn.Linear(low_dim_total, low_dim_feat_dim), nn.ReLU(),
            nn.Linear(low_dim_feat_dim, low_dim_feat_dim))
        self.low_token_proj = (nn.Linear(low_dim_feat_dim, self.token_dim)
                               if self.use_resampler else None)

        self.fuse = nn.Sequential(
            nn.Linear(n_views * view_feat_dim + low_dim_feat_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)) if not self.use_resampler else None

        # 供门控/分析使用的缓存（v2/v3 会用到）
        self.last_patch_tokens: Dict[str, torch.Tensor] = {}

    # ---- 供官方脚手架调用 ----
    def output_shape(self):
        return [self.token_dim]

    @property
    def tokens_per_obs(self) -> int:
        if self.use_resampler:
            return 1 + self.n_views * self.resampler_num_queries
        return 1

    def _encode_views(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """每个视角独立编码，**共享同一套 DINOv2 权重**（CoRP 的做法）。"""
        B = obs_dict[self.rgb_keys[0]].shape[0]
        V = len(self.rgb_keys)
        imgs = []
        for k in self.rgb_keys:
            x = obs_dict[k]                                     # (B, 3, H, W)
            x = self.crop(x)                                    # → 224（14 的倍数）
            x = (x - self.img_mean.to(x)) / self.img_std.to(x)  # ImageNet 归一化
            imgs.append(x)
        # 把 V 个视角并到 batch 维，一次前向（共享权重）
        cat = torch.stack(imgs, dim=1).reshape(B * V, *imgs[0].shape[1:])
        feats = self.dino.forward_features(cat)["x_norm_patchtokens"]   # (B*V, T, d)
        if self.use_resampler:
            lang = obs_dict.get("lang_emb", None)
            if lang is not None:
                lang = lang[:, None, :].expand(B, V, lang.shape[-1]).reshape(B * V, -1)
            tokens = self.resampler(feats, lang)                # (B*V, K, token_dim)
            tokens = tokens.reshape(B, V, self.resampler_num_queries, self.token_dim)
            if self.token_style is not None:
                tokens = self.token_style(
                    tokens, domain_ids=self.current_domain_ids,
                    training=self.training)
            self.last_patch_tokens = {
                "resampled_tokens": tokens.detach(),
                "patch_tokens": feats.detach(),
                "resampled": True,
            }
            return tokens
        if self.pool == "meanmax":
            pooled = torch.cat([feats.mean(dim=1), feats.amax(dim=1)], dim=-1)
        else:
            pooled = feats.mean(dim=1)                          # (B*V, d)
        pf = pooled.reshape(B, V, -1)                           # (B, V, view_in)
        self.last_patch_tokens = {
            "pooled": pf.detach(),
            "patch_tokens": feats.detach(),
            "resampled": False,
        }
        return pf

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        vt = self._encode_views(obs_dict)
        low = torch.cat([obs_dict[k] for k in self.low_dim_keys], dim=-1)
        lf = self.low_dim_net(low)
        if self.use_resampler:
            # One low-dim/language token + K tokens per camera view.
            low_tok = self.low_token_proj(lf).unsqueeze(1)       # (B,1,D)
            visual = vt.reshape(vt.shape[0], -1, self.token_dim) # (B,V*K,D)
            return torch.cat([low_tok, visual], dim=1)            # (B,1+V*K,D)
        vf = self.view_proj(vt).reshape(vt.shape[0], -1)
        return self.fuse(torch.cat([vf, lf], dim=-1))
