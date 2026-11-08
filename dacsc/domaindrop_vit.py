"""DomainDrop-style channel dropout for the CoRP DINOv2 encoder.

The channel space is the 384-wide output of a DINOv2 transformer block:

    [batch, views, tokens, channels]

The official DomainDrop implementation operates on CNN feature maps. This
module keeps the channel-score, weighted-random-selection, and recovery
semantics from the official implementation while adapting the tensor layout
to CoRP's view-major visual encoder.

Source references:
    https://github.com/lingeringlight/DomainDrop
    models/FilterDropout.py
    models/LayerDiscriminator.py

The official repository has no ViT implementation and no explicit license file.
This is an attributed project adaptation, not an official DomainDrop release.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

CHANNEL_METHODS = ("none", "random", "domaindrop_vit")
DOMAINDROP_REVISION = "a2d9607d4f1fb9ba35a9fa48fbceff82fd67d15f"


@dataclass(frozen=True)
class ChannelDropoutConfig:
    method: str = "none"
    layers: tuple[int, ...] = (2, 5, 8, 11)
    drop_rate: float = 0.33
    layer_probability: float = 0.8
    domain_loss_weight: float = 1.0
    consistency_weight: float = 1.5
    grl_lambda: float = 0.25
    views_shared: bool = True
    wrs: bool = True
    recover: bool = True

    def __post_init__(self) -> None:
        if self.method not in CHANNEL_METHODS:
            raise ValueError(f"unknown channel method: {self.method}")
        if (
            not self.layers
            or len(self.layers) != len(set(self.layers))
            or any(type(layer) is not int or layer < 0 for layer in self.layers)
        ):
            raise ValueError("channel layers must be unique nonnegative integers")
        if not math.isfinite(self.drop_rate) or not 0 < self.drop_rate < 1:
            raise ValueError("drop_rate must be in (0,1)")
        if (
            not math.isfinite(self.layer_probability)
            or not 0 <= self.layer_probability <= 1
        ):
            raise ValueError("layer_probability must be in [0,1]")
        if (
            not math.isfinite(self.domain_loss_weight)
            or self.domain_loss_weight < 0
            or not math.isfinite(self.consistency_weight)
            or self.consistency_weight < 0
            or not math.isfinite(self.grl_lambda)
            or self.grl_lambda < 0
        ):
            raise ValueError("loss weights and GRL lambda must be nonnegative")
        if not self.views_shared:
            raise ValueError(
                "the first DomainDrop-ViT implementation requires shared-view masks"
            )

    def contract(self) -> dict:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        payload["provenance"] = {
            "source": "DomainDrop official code + CoRP multi-view adaptation",
            "official_repository": "https://github.com/lingeringlight/DomainDrop",
            "official_revision": DOMAINDROP_REVISION,
            "official_vit_available": False,
            "official_channel_space": "CNN feature map [B,C,H,W]",
            "corp_channel_space": "ViT block output [B,C,views,tokens]",
            "corp_mask_scope": "shared over views and tokens",
            "selection": "source-domain classifier weights + weighted random selection",
            "recovery": "official mask scale recovery",
        }
        return payload


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, lambd):
        ctx.lambd = lambd
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.lambd, None


def grad_reverse(value: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverse.apply(value, lambd)


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    flat = scores.flatten(1)
    score_max = flat.max(dim=1, keepdim=True)[0]
    score_min = flat.min(dim=1, keepdim=True)[0]
    scale = score_max - score_min
    normalized = (flat - score_min) / scale.clamp_min(1e-12)
    normalized = normalized.view_as(scores)
    # The official implementation leaves a constant map undefined. A constant
    # score has no domain-sensitive direction, so its normalized score is zero.
    positive = scale.view(-1, *([1] * (scores.ndim - 1))) > 0
    return torch.where(positive, normalized, torch.zeros_like(scores))


def _weighted_random_mask(
    scores: torch.Tensor,
    drop_rate: float,
    *,
    wrs: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select channels using the official weighted-random-selection formula."""
    batch_size, channels = scores.shape
    drop_num = int(channels * drop_rate)
    if not 0 <= drop_num < channels:
        raise ValueError("drop_rate selects an invalid number of channels")
    random_values = torch.rand(
        scores.shape,
        device=scores.device,
        dtype=scores.dtype,
        generator=generator,
    )
    if wrs:
        normalized = _normalize_scores(scores[:, :, None, None])[:, :, 0, 0]
        keys = random_values.pow(1.0 / normalized.clamp_min(1e-8))
    else:
        keys = random_values
    if drop_num:
        threshold = torch.sort(keys, dim=1, descending=True)[0][:, drop_num]
        selected = keys > threshold[:, None]
    else:
        selected = torch.zeros_like(keys, dtype=torch.bool)
    return (~selected).to(dtype=scores.dtype)


def _random_channel_mask(
    batch_size: int,
    channels: int,
    drop_rate: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Matched random control with exactly the same number of dropped channels."""
    drop_num = int(channels * drop_rate)
    scores = torch.rand(
        (batch_size, channels),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    indices = torch.topk(scores, k=drop_num, dim=1, largest=False).indices
    mask = torch.ones_like(scores)
    if drop_num:
        mask.scatter_(1, indices, 0)
    return mask


def _recover_scale(mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        raise ValueError("channel mask would zero all channels")
    return mask * mask.numel() / mask.sum()


class LayerDomainDrop(nn.Module):
    """Official DomainDrop-style channel score and mask for one ViT block."""

    def __init__(
        self,
        channels: int,
        num_domains: int,
        *,
        grl_lambda: float,
        wrs: bool,
        recover: bool,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_domains = num_domains
        self.grl_lambda = grl_lambda
        self.wrs = wrs
        self.recover = recover
        self.model = nn.Linear(channels, num_domains)

    def forward(
        self,
        feature_map: torch.Tensor,
        domain_labels: torch.Tensor,
        drop_rate: float,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_map.ndim != 4 or feature_map.shape[1] != self.channels:
            raise ValueError("feature_map must be [batch, channels, views, tokens]")
        if domain_labels.ndim != 1 or domain_labels.shape[0] != feature_map.shape[0]:
            raise ValueError("domain labels must match the feature-map batch")
        pooled = feature_map.mean(dim=(2, 3))
        logits = self.model(grad_reverse(pooled, self.grl_lambda))

        # DomainDrop's get_scores uses the classifier row for each source domain.
        weights = self.model.weight.detach().to(
            device=feature_map.device, dtype=feature_map.dtype
        )
        domain_weight = weights[domain_labels].view(
            feature_map.shape[0], self.channels, 1, 1
        )
        scores = feature_map.detach() * domain_weight
        scores = _normalize_scores(scores)
        channel_scores = F.adaptive_avg_pool2d(scores, 1).flatten(1)
        mask = _weighted_random_mask(
            channel_scores,
            drop_rate,
            wrs=self.wrs,
            generator=generator,
        )
        if self.recover:
            mask = _recover_scale(mask)
        return logits, mask


class ChannelDropoutController:
    """Train-only hooks over selected CoRP visual blocks.

    The model receives concatenated view-major images. A block output is
    therefore `[views * batch, tokens, channels]`; this controller reshapes it
    to `[batch, channels, views, tokens]` for a shared per-observation mask.
    """

    def __init__(
        self,
        policy,
        config: ChannelDropoutConfig,
        style_to_id: dict[int, int],
    ) -> None:
        self.policy = policy
        self.config = config
        self.style_to_id = dict(style_to_id)
        if config.method == "none":
            raise ValueError("none channel method does not need a controller")
        backbone = policy.representation_extractor.visual_encoder.backbone
        if len(config.layers) > len(backbone.blocks):
            raise ValueError("channel layer exceeds the visual encoder depth")
        if not self.style_to_id:
            raise ValueError("style_to_id cannot be empty")
        self.num_domains = len(self.style_to_id)
        self._training = True
        self._active_layer: int | None = None
        self._style_labels: torch.Tensor | None = None
        self._domain_logits: list[tuple[int, torch.Tensor]] = []
        self._masks: list[tuple[int, torch.Tensor]] = []
        self._handles = []
        self.discriminators = nn.ModuleDict(
            {
                str(layer): LayerDomainDrop(
                    int(policy.representation_extractor.visual_encoder.backbone.blocks[
                        layer
                    ].mlp.fc1.in_features),
                    self.num_domains,
                    grl_lambda=config.grl_lambda,
                    wrs=config.wrs,
                    recover=config.recover,
                )
                for layer in config.layers
            }
        )
        self.discriminators.to(
            device=next(policy.parameters()).device,
            dtype=next(policy.parameters()).dtype,
        )
        for layer in config.layers:
            self._handles.append(
                backbone.blocks[layer].register_forward_hook(
                    self._make_hook(layer)
                )
            )

    def parameters(self):
        return self.discriminators.parameters()

    def state_dict(self) -> dict:
        return {
            "config": self.config.contract(),
            "style_to_id": {str(key): value for key, value in self.style_to_id.items()},
            "discriminators": self.discriminators.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state["config"] != self.config.contract():
            raise ValueError("channel dropout configuration differs from checkpoint")
        expected_styles = {str(key): value for key, value in self.style_to_id.items()}
        if state["style_to_id"] != expected_styles:
            raise ValueError("style-to-domain mapping differs from checkpoint")
        self.discriminators.load_state_dict(state["discriminators"], strict=True)

    def train(self) -> "ChannelDropoutController":
        self._training = True
        self.discriminators.train()
        return self

    def eval(self) -> "ChannelDropoutController":
        self._training = False
        self.discriminators.eval()
        return self

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def begin_step(self, style_ids: torch.Tensor) -> None:
        if style_ids.ndim != 1:
            raise ValueError("style_ids must be one-dimensional")
        mapped = []
        for value in style_ids.detach().cpu().tolist():
            if value not in self.style_to_id:
                raise ValueError(f"style {value} is outside the source domain map")
            mapped.append(self.style_to_id[value])
        self._style_labels = torch.tensor(
            mapped, dtype=torch.long, device=style_ids.device
        )
        self._active_layer = None
        if random.random() <= self.config.layer_probability:
            self._active_layer = random.choice(self.config.layers)
        self._domain_logits = []
        self._masks = []

    @property
    def domain_logits(self) -> list[tuple[int, torch.Tensor]]:
        return self._domain_logits

    @property
    def masks(self) -> list[tuple[int, torch.Tensor]]:
        return self._masks

    @property
    def active_layer(self) -> int | None:
        return self._active_layer

    @property
    def domain_labels(self) -> torch.Tensor | None:
        return self._style_labels

    @torch.no_grad()
    def diagnostics(self) -> dict:
        """Return detached training diagnostics for logging and gates."""
        payload: dict = {
            "active_layer": self._active_layer,
            "configured_layers": list(self.config.layers),
            "drop_rate": self.config.drop_rate,
            "layer_probability": self.config.layer_probability,
            "mask_events": [],
            "domain_accuracy": None,
            "domain_balanced_accuracy": None,
            "domain_per_style_recall": {},
            "domain_top_confusions": [],
        }
        for layer, mask in self._masks:
            dropped = (mask == 0).sum(dim=1).to(dtype=torch.float32)
            payload["mask_events"].append(
                {
                    "layer": layer,
                    "samples": int(mask.shape[0]),
                    "channels": int(mask.shape[1]),
                    "nominal_drop_fraction": float(self.config.drop_rate),
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

        if not self._domain_logits or self._style_labels is None:
            return payload

        logits = torch.stack(
            [layer_logits.detach() for _, layer_logits in self._domain_logits],
            dim=0,
        ).mean(dim=0)
        labels = self._style_labels.detach()
        predictions = logits.argmax(dim=1)
        correct = predictions.eq(labels)
        payload["domain_accuracy"] = float(correct.float().mean())

        recalls = {}
        for class_id in labels.unique(sorted=True).tolist():
            class_mask = labels.eq(class_id)
            recalls[str(class_id)] = float(correct[class_mask].float().mean())
        payload["domain_per_style_recall"] = recalls
        if recalls:
            payload["domain_balanced_accuracy"] = sum(recalls.values()) / len(recalls)

        num_domains = self.num_domains
        confusion = torch.bincount(
            labels * num_domains + predictions,
            minlength=num_domains * num_domains,
        ).reshape(num_domains, num_domains)
        off_diagonal = confusion.clone()
        off_diagonal.fill_diagonal_(0)
        values, indices = off_diagonal.flatten().topk(
            k=min(5, int((off_diagonal > 0).sum()))
        )
        payload["domain_top_confusions"] = [
            {
                "true": int(index // num_domains),
                "predicted": int(index % num_domains),
                "count": int(value),
            }
            for value, index in zip(values.tolist(), indices.tolist(), strict=True)
            if value > 0
        ]
        return payload

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            if not self._training or not self.policy.training:
                return output
            if self._style_labels is None:
                raise RuntimeError("begin_step must run before the policy forward")
            views = len(self.policy.view_names)
            batch_size = self._style_labels.shape[0]
            if output.ndim != 3 or output.shape[0] != views * batch_size:
                raise ValueError(
                    "visual block output must be [views*batch, tokens, channels]"
                )
            _, tokens, channels = output.shape
            feature_map = (
                output.reshape(views, batch_size, tokens, channels)
                .permute(1, 3, 0, 2)
                .contiguous()
            )
            if self.config.method == "domaindrop_vit":
                discriminator = self.discriminators[str(layer)]
                logits, score_mask = discriminator(
                    feature_map,
                    self._style_labels,
                    self.config.drop_rate,
                )
                self._domain_logits.append((layer, logits))
                if layer != self._active_layer:
                    return output
                mask = score_mask
            else:
                discriminator = self.discriminators[str(layer)]
                logits, _ = discriminator(
                    feature_map,
                    self._style_labels,
                    self.config.drop_rate,
                )
                self._domain_logits.append((layer, logits))
                if layer != self._active_layer:
                    return output
                mask = _random_channel_mask(
                    batch_size,
                    channels,
                    self.config.drop_rate,
                    device=output.device,
                    dtype=output.dtype,
                )
                if self.config.recover:
                    mask = _recover_scale(mask)
            self._masks.append((layer, mask))
            if layer != self._active_layer:
                return output
            masked = feature_map * mask[:, :, None, None]
            return (
                masked.permute(2, 0, 3, 1)
                .contiguous()
                .view(views * batch_size, tokens, channels)
            )

        return hook
