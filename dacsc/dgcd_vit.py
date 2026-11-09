"""DgCD discrepancy-guided channel dropout for the CoRP DINOv2 encoder.

Official DgCD operates on CNN feature maps and assumes fixed groups of 32
samples per environment. This module preserves its environment-wise alignment,
discrepancy scoring, weighted random selection, and recovery while adapting
the tensor layout to CoRP's view-major ViT encoder.

Source references:
    https://github.com/gyeomo/DgCD
    domainbed/networks/DgCD.py

The official repository has no ViT implementation. This is an attributed
project adaptation, not an official DgCD ViT release.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

import torch
from torch.nn import functional as F

DGCD_METHODS = ("dgcd_random", "dgcd")
DGCD_REVISION = "56f65c33f1f61a4539d65784f2ecc2c8c299eb61"


@dataclass(frozen=True)
class DgCDConfig:
    method: str = "dgcd_random"
    layers: tuple[int, ...] = (2, 5, 8, 11)
    drop_rate: float = 0.33
    rho: float = 0.05
    group_size: int = 32
    groups_per_batch: int = 3
    views_shared: bool = True
    recover: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.method not in DGCD_METHODS:
            raise ValueError(f"unknown DgCD method: {self.method}")
        if (
            not self.layers
            or len(self.layers) < 2
            or len(self.layers) != len(set(self.layers))
            or any(type(layer) is not int or layer < 0 for layer in self.layers)
        ):
            raise ValueError("DgCD requires at least two unique nonnegative layers")
        if not math.isfinite(self.drop_rate) or not 0 < self.drop_rate < 1:
            raise ValueError("DgCD drop_rate must be in (0,1)")
        if not math.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("DgCD rho must be positive")
        if type(self.group_size) is not int or self.group_size <= 0:
            raise ValueError("DgCD group_size must be a positive integer")
        if type(self.groups_per_batch) is not int or self.groups_per_batch <= 0:
            raise ValueError("DgCD groups_per_batch must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("DgCD seed must be an integer")
        if not self.views_shared:
            raise ValueError("the first DgCD-ViT implementation requires shared-view masks")

    @property
    def batch_size(self) -> int:
        return self.group_size * self.groups_per_batch

    def contract(self) -> dict:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        payload["batch_size"] = self.batch_size
        payload["provenance"] = {
            "source": "DgCD official code + CoRP multi-view adaptation",
            "official_repository": "https://github.com/gyeomo/DgCD",
            "official_revision": DGCD_REVISION,
            "official_vit_available": False,
            "official_channel_space": "CNN feature map [B,C,H,W]",
            "corp_channel_space": "ViT block output [B,C,views,tokens]",
            "corp_mask_scope": "shared over views and tokens",
            "environment_batch": "single-style contiguous groups of fixed size",
            "score_dtype": "float32",
            "recovery": "official mask numel/kept-count scale recovery",
        }
        return payload


def _scaling(value: torch.Tensor) -> torch.Tensor:
    value_max = value.max(dim=-1, keepdim=True)[0]
    value_min = value.min(dim=-1, keepdim=True)[0]
    scale = value_max - value_min
    normalized = (value - value_min) / scale.clamp_min(1e-12)
    return torch.where(scale > 0, normalized, torch.zeros_like(value))


def _update_scaler(parameter: torch.Tensor, rho: float) -> None:
    if parameter.grad is None:
        raise RuntimeError("DgCD local scaler did not receive a gradient")
    grad = parameter.grad
    step = rho * grad.div(torch.sqrt(torch.norm(grad, dim=-1, keepdim=True) + 1e-12))
    parameter.data.add_(step)


def _alignment(
    values: torch.Tensor,
    scaler_total_mean: torch.Tensor,
    scaler_total_scale: torch.Tensor,
    scaler_env_mean: list[torch.Tensor],
    scaler_env_scale: list[torch.Tensor],
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_mean = values.mean(dim=0, keepdim=True)
    total_var = values.var(dim=0, keepdim=True, unbiased=False)
    aligned_total = (values - total_mean) / torch.sqrt(total_var + 1e-5)
    aligned_total = aligned_total * scaler_total_scale + scaler_total_mean

    env_values = []
    for group_index, start in enumerate(range(0, values.shape[0], group_size)):
        group = values[start : start + group_size]
        group_mean = group.mean(dim=0, keepdim=True)
        group_var = group.var(dim=0, keepdim=True, unbiased=False)
        aligned = (group - group_mean) / torch.sqrt(group_var + 1e-5)
        aligned = (
            aligned * scaler_env_scale[group_index]
            + scaler_env_mean[group_index]
        )
        env_values.append(aligned)
    return torch.cat(env_values, dim=0), aligned_total


@torch.no_grad()
def _weighted_random_mask(
    scores: torch.Tensor,
    drop_rate: float,
    *,
    rng: random.Random,
) -> torch.Tensor:
    channels = scores.shape[1]
    drop_num = int(channels * drop_rate)
    random_values = torch.tensor(
        [[rng.random() for _ in range(channels)] for _ in range(scores.shape[0])],
        device=scores.device,
        dtype=scores.dtype,
    )
    keys = random_values.pow(1.0 / scores.clamp_min(1e-8))
    if drop_num:
        threshold = torch.sort(keys, dim=1, descending=True)[0][:, drop_num]
        selected = keys > threshold[:, None]
    else:
        selected = torch.zeros_like(keys, dtype=torch.bool)
    mask = (~selected).to(dtype=scores.dtype)
    if not mask.any():
        raise ValueError("DgCD mask would zero all channels")
    return mask * mask.numel() / mask.sum()


def dgcd_scores(
    feature_map: torch.Tensor,
    *,
    rho: float,
    group_size: int,
) -> torch.Tensor:
    """Return official-style DgCD channel discrepancy scores in FP32."""
    if feature_map.ndim != 4:
        raise ValueError("feature_map must be [batch, channels, views, tokens]")
    if feature_map.shape[0] % group_size:
        raise ValueError("DgCD batch must contain complete environment groups")
    values = feature_map.detach().float().mean(dim=(2, 3))
    batch_size, channels = values.shape
    num_groups = batch_size // group_size

    total_mean = torch.zeros(
        (batch_size, channels), device=values.device, dtype=values.dtype
    ).requires_grad_(True)
    total_scale = torch.ones_like(total_mean).requires_grad_(True)
    env_mean = [
        torch.zeros(
            (group_size, channels), device=values.device, dtype=values.dtype
        ).requires_grad_(True)
        for _ in range(num_groups)
    ]
    env_scale = [
        torch.ones(
            (group_size, channels), device=values.device, dtype=values.dtype
        ).requires_grad_(True)
        for _ in range(num_groups)
    ]

    aligned_env, aligned_total = _alignment(
        values,
        total_mean,
        total_scale,
        env_mean,
        env_scale,
        group_size,
    )
    loss = F.kl_div(
        F.log_softmax(aligned_total, dim=1),
        F.log_softmax(aligned_env, dim=1),
        reduction="batchmean",
        log_target=True,
    )
    loss.backward()
    for parameter in [total_mean, total_scale, *env_mean, *env_scale]:
        _update_scaler(parameter, rho)

    with torch.no_grad():
        aligned_env, aligned_total = _alignment(
            values,
            total_mean.detach(),
            total_scale.detach(),
            [value.detach() for value in env_mean],
            [value.detach() for value in env_scale],
            group_size,
        )
        gram = torch.mm(
            aligned_total.t() + 1e-7,
            aligned_env + 1e-7,
        ).diag()
        aligned_total = _scaling(aligned_total / gram)
        aligned_env = _scaling(aligned_env / gram)
        scores = _scaling((aligned_total - aligned_env).pow(2))
    return scores


class DgCDController:
    """Train-only DgCD hooks over selected CoRP ViT blocks."""

    def __init__(self, policy, config: DgCDConfig) -> None:
        self.policy = policy
        self.config = config
        backbone = policy.representation_extractor.visual_encoder.backbone
        if len(config.layers) > len(backbone.blocks):
            raise ValueError("DgCD layer exceeds the visual encoder depth")
        self._training = True
        self._step = 0
        self._active_layers: tuple[int, ...] = ()
        self._masks: list[tuple[int, torch.Tensor]] = []
        self._scores: list[tuple[int, torch.Tensor]] = []
        self._handles = [
            backbone.blocks[layer].register_forward_hook(self._make_hook(layer))
            for layer in config.layers
        ]

    def state_dict(self) -> dict:
        return {"config": self.config.contract()}

    def load_state_dict(self, state: dict) -> None:
        if state["config"] != self.config.contract():
            raise ValueError("DgCD configuration differs from checkpoint")

    def train(self) -> "DgCDController":
        self._training = True
        return self

    def eval(self) -> "DgCDController":
        self._training = False
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def begin_step(self, step: int) -> None:
        if type(step) is not int or step < 0:
            raise ValueError("DgCD step must be a nonnegative integer")
        self._step = step
        rng = random.Random(self.config.seed + step * 1_000_003)
        random_layer = rng.choice(self.config.layers[:-1])
        self._active_layers = tuple(
            sorted({random_layer, self.config.layers[-1]})
        )
        self._masks = []
        self._scores = []

    @property
    def active_layers(self) -> tuple[int, ...]:
        return self._active_layers

    @property
    def masks(self) -> list[tuple[int, torch.Tensor]]:
        return self._masks

    def diagnostics(self) -> dict:
        return {
            "method": self.config.method,
            "step": self._step,
            "active_layers": list(self._active_layers),
            "group_size": self.config.group_size,
            "groups_per_batch": self.config.groups_per_batch,
            "mask_events": [
                {
                    "layer": layer,
                    "samples": int(mask.shape[0]),
                    "channels": int(mask.shape[1]),
                    "mean_actual_drop_fraction": float(
                        (mask == 0).float().mean()
                    ),
                    "mean_kept_fraction_with_recovery": float(mask.mean()),
                }
                for layer, mask in self._masks
            ],
            "score_events": [
                {
                    "layer": layer,
                    "mean": float(score.mean()),
                    "std": float(score.std(unbiased=False)),
                    "min": float(score.min()),
                    "max": float(score.max()),
                }
                for layer, score in self._scores
            ],
        }

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            if not self._training or not self.policy.training:
                return output
            if layer not in self._active_layers:
                return output
            views = len(self.policy.view_names)
            if output.ndim != 3 or output.shape[0] % views:
                raise ValueError(
                    "visual block output must be [views*batch, tokens, channels]"
                )
            batch_size = output.shape[0] // views
            if batch_size != self.config.batch_size:
                raise ValueError(
                    f"DgCD requires batch {self.config.batch_size}, got {batch_size}"
                )
            _, tokens, channels = output.shape
            feature_map = (
                output.reshape(views, batch_size, tokens, channels)
                .permute(1, 3, 0, 2)
                .contiguous()
            )
            rng = random.Random(
                self.config.seed + self._step * 7_919 + layer * 104_729
            )
            if self.config.method == "dgcd":
                scores = dgcd_scores(
                    feature_map,
                    rho=self.config.rho,
                    group_size=self.config.group_size,
                )
            else:
                scores = torch.tensor(
                    [
                        [rng.random() for _ in range(channels)]
                        for _ in range(batch_size)
                    ],
                    device=feature_map.device,
                    dtype=torch.float32,
                )
            self._scores.append((layer, scores.detach()))
            mask = _weighted_random_mask(
                scores,
                self.config.drop_rate,
                rng=rng,
            ).to(dtype=output.dtype)
            self._masks.append((layer, mask))
            masked = feature_map * mask[:, :, None, None]
            return masked.permute(2, 0, 3, 1).reshape(
                views * batch_size, tokens, channels
            )

        return hook
