"""PLACE progressive channel dropout for the CoRP DINOv2 encoder.

Official PLACE operates on CNN feature maps. This module preserves its
layer-wise random dropout, progressive dropout ratio, and recovery formula
while adapting the tensor layout to CoRP's view-major ViT encoder.

Source references:
    https://github.com/lingeringlight/PLACEdropout
    models/resnet_PLACE.py

The official repository has no ViT implementation. This is an attributed
project adaptation, not an official PLACE ViT release.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

import torch

PLACE_METHODS = ("place_random", "place")
PLACE_REVISION = "d17f0a79dc2795f315a68ab745aa98b231000f2b"


@dataclass(frozen=True)
class PlaceDropoutConfig:
    method: str = "place_random"
    layers: tuple[int, ...] = (2, 5, 8, 11)
    drop_rate: float = 0.33
    velocity: float = 4.0
    reference_epochs: int = 30
    views_shared: bool = True
    recover: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.method not in PLACE_METHODS:
            raise ValueError(f"unknown PLACE method: {self.method}")
        if (
            not self.layers
            or len(self.layers) != len(set(self.layers))
            or any(type(layer) is not int or layer < 0 for layer in self.layers)
        ):
            raise ValueError("PLACE layers must be unique nonnegative integers")
        if not math.isfinite(self.drop_rate) or not 0 < self.drop_rate < 1:
            raise ValueError("PLACE drop_rate must be in (0,1)")
        if not math.isfinite(self.velocity) or self.velocity <= 0:
            raise ValueError("PLACE velocity must be positive")
        if type(self.reference_epochs) is not int or self.reference_epochs <= 0:
            raise ValueError("PLACE reference_epochs must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("PLACE seed must be an integer")
        if not self.views_shared:
            raise ValueError("the first PLACE-ViT implementation requires shared-view masks")

    def drop_fraction(self, step: int, total_steps: int) -> float:
        if type(step) is not int or step < 0:
            raise ValueError("PLACE step must be a nonnegative integer")
        if type(total_steps) is not int or total_steps <= 0 or step > total_steps:
            raise ValueError("PLACE total_steps must cover the current step")
        if self.method == "place_random":
            return self.drop_rate
        equivalent_epoch = self.reference_epochs * step / total_steps
        return (
            self.drop_rate
            * 2.0
            / math.pi
            * math.atan(equivalent_epoch / self.velocity)
        )

    def contract(self) -> dict:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        payload["provenance"] = {
            "source": "PLACE official code + CoRP multi-view adaptation",
            "official_repository": "https://github.com/lingeringlight/PLACEdropout",
            "official_revision": PLACE_REVISION,
            "official_vit_available": False,
            "official_channel_space": "CNN feature map [B,C,H,W]",
            "corp_channel_space": "ViT block output [B,C,views,tokens]",
            "corp_mask_scope": "shared over views and tokens",
            "progressive_progress": "equivalent official stage-2 epochs",
            "recovery": "official 1/(1-p) scale recovery",
        }
        return payload


def _random_channel_mask(
    batch_size: int,
    channels: int,
    drop_fraction: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rng: random.Random,
) -> torch.Tensor:
    drop_num = int(channels * drop_fraction)
    mask = torch.ones((batch_size, channels), device=device, dtype=dtype)
    for batch_index in range(batch_size):
        if drop_num:
            dropped = rng.sample(range(channels), drop_num)
            mask[batch_index, dropped] = 0
    return mask


class PlaceDropoutController:
    """Train-only PLACE hooks over selected CoRP ViT blocks."""

    def __init__(self, policy, config: PlaceDropoutConfig) -> None:
        self.policy = policy
        self.config = config
        backbone = policy.representation_extractor.visual_encoder.backbone
        if len(config.layers) > len(backbone.blocks):
            raise ValueError("PLACE layer exceeds the visual encoder depth")
        self._training = True
        self._step = 0
        self._total_steps = 0
        self._active_layer: int | None = None
        self._drop_fraction = 0.0
        self._masks: list[tuple[int, torch.Tensor]] = []
        self._handles = [
            backbone.blocks[layer].register_forward_hook(self._make_hook(layer))
            for layer in config.layers
        ]

    def state_dict(self) -> dict:
        return {"config": self.config.contract()}

    def load_state_dict(self, state: dict) -> None:
        if state["config"] != self.config.contract():
            raise ValueError("PLACE configuration differs from checkpoint")

    def train(self) -> "PlaceDropoutController":
        self._training = True
        return self

    def eval(self) -> "PlaceDropoutController":
        self._training = False
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def begin_step(self, step: int, total_steps: int) -> None:
        self._step = step
        self._total_steps = total_steps
        self._drop_fraction = self.config.drop_fraction(step, total_steps)
        layer_rng = random.Random(self.config.seed + step * 1_000_003)
        self._active_layer = layer_rng.choice(self.config.layers)
        self._masks = []

    @property
    def active_layer(self) -> int | None:
        return self._active_layer

    @property
    def drop_fraction(self) -> float:
        return self._drop_fraction

    @property
    def masks(self) -> list[tuple[int, torch.Tensor]]:
        return self._masks

    def diagnostics(self) -> dict:
        payload = {
            "method": self.config.method,
            "step": self._step,
            "total_steps": self._total_steps,
            "active_layer": self._active_layer,
            "drop_fraction": self._drop_fraction,
            "recover_factor": (
                1.0 / (1.0 - self._drop_fraction) if self.config.recover else 1.0
            ),
            "mask_events": [],
        }
        for layer, mask in self._masks:
            dropped = (mask == 0).sum(dim=1).to(dtype=torch.float32)
            payload["mask_events"].append(
                {
                    "layer": layer,
                    "samples": int(mask.shape[0]),
                    "channels": int(mask.shape[1]),
                    "mean_actual_drop_fraction": float(
                        dropped.mean() / mask.shape[1]
                    ),
                    "min_actual_drop_fraction": float(
                        dropped.min() / mask.shape[1]
                    ),
                    "max_actual_drop_fraction": float(
                        dropped.max() / mask.shape[1]
                    ),
                    "mean_kept_fraction": float(mask.mean()),
                }
            )
        return payload

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            if not self._training or not self.policy.training:
                return output
            if layer != self._active_layer:
                return output
            views = len(self.policy.view_names)
            if output.ndim != 3 or output.shape[0] % views:
                raise ValueError(
                    "visual block output must be [views*batch, tokens, channels]"
                )
            batch_size = output.shape[0] // views
            _, tokens, channels = output.shape
            rng = random.Random(
                self.config.seed + self._step * 7_919 + layer * 104_729
            )
            mask = _random_channel_mask(
                batch_size,
                channels,
                self._drop_fraction,
                device=output.device,
                dtype=output.dtype,
                rng=rng,
            )
            self._masks.append((layer, mask))
            scale = (
                1.0 / (1.0 - self._drop_fraction) if self.config.recover else 1.0
            )
            feature_map = output.reshape(views, batch_size, tokens, channels)
            masked = feature_map * (
                mask.reshape(1, batch_size, 1, channels) * scale
            )
            return masked.reshape(views * batch_size, tokens, channels)

        return hook
