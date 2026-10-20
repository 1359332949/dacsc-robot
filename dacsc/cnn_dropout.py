"""Faithful CNN channel-dropout variants for the official ResNet18 baseline.

Official DomainDrop structure implemented here:
  * one domain discriminator per ResNet stage-end feature map;
  * all domain heads receive supervision every step;
  * one stage is randomly selected for masking (``layer_wise_prob``);
  * only the selected stage receives the WRS channel mask.

PLACE and DgCD are also implemented on the same CNN feature maps.  The module
works on ``ResNet18ConvFiLM`` FiLM outputs of shape ``(B, C, H, W)``.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dacsc.dg_methods import DgCDScorer, grad_reverse
from dacsc.hard_gate import wrs_mask


class CNNChannelDropoutController(nn.Module):
    MODES = ("none", "place", "domaindrop", "domaindrop_norm", "dgcd")

    def __init__(self, film_layers: Sequence[nn.Module], mode: str = "none",
                 layer_selection: str = "fixed",
                 layer_indices: Sequence[int] | None = None,
                 drop_ratio: float = 0.33, recover: bool = True,
                 progressive: bool = False, velocity: float = 50.0,
                 layer_wise_prob: float = 1.0,
                 num_domains: int = 10, domain_lambda: float = 1.0,
                 domain_grl: float = 0.25, grl_warmup_steps: int = 10,
                 domain_head_norm: bool = False, train_only: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown cnn_dropout_mode={mode}")
        self.mode = str(mode)
        self.layer_selection = str(layer_selection)
        self.layer_indices = (None if layer_indices is None
                              else [int(i) for i in layer_indices])
        self.drop_ratio = float(drop_ratio)
        self.recover = bool(recover)
        self.progressive = bool(progressive)
        self.velocity = float(velocity)
        self.layer_wise_prob = float(layer_wise_prob)
        self.num_domains = int(num_domains)
        self.domain_lambda = float(domain_lambda)
        self.domain_grl = float(domain_grl)
        self.grl_warmup_steps = max(int(grl_warmup_steps), 1)
        self.domain_head_norm = bool(domain_head_norm) or mode == "domaindrop_norm"
        self.train_only = bool(train_only)
        self.eps = float(eps)

        self.channels = [int(layer.lang_proj.out_features) // 2
                         for layer in film_layers]
        self.n_layers = len(self.channels)
        # ResNet18 is split into two basic blocks per stage; the stage-end
        # features are the last FiLM output of each stage.
        self.stage_end_indices = [
            i for i in range(self.n_layers)
            if i == self.n_layers - 1
            or self.channels[i] != self.channels[i + 1]
        ]
        self.domain_layers = self._domain_layer_indices()

        self.domain_heads = nn.ModuleDict()
        if self.mode in ("domaindrop", "domaindrop_norm"):
            for i in self.domain_layers:
                c = self.channels[i]
                base = nn.Linear(c, self.num_domains)
                head = (nn.Sequential(nn.LayerNorm(c), base)
                        if self.domain_head_norm else base)
                self.domain_heads[str(i)] = head
        if self.mode == "dgcd":
            self.dgcd = nn.ModuleDict({
                str(i): DgCDScorer() for i in range(self.n_layers)})

        self.domain_ids: torch.Tensor | None = None
        self.step = 0
        self._aux: List[torch.Tensor] = []
        self._metrics: Dict[str, float] = {}
        self._active_layers: List[int] | None = None
        self._do_mask: bool | None = None

    def _domain_layer_indices(self) -> List[int]:
        if self.mode not in ("domaindrop", "domaindrop_norm"):
            return []
        if self.layer_selection == "fixed":
            ids = self.layer_indices or [self.n_layers - 1]
            return [int(i) for i in ids if 0 <= int(i) < self.n_layers]
        # Official DomainDrop supervises every ResNet stage discriminator.
        return list(self.stage_end_indices)

    def set_context(self, domain_ids: torch.Tensor | None, step: int) -> None:
        self.domain_ids = domain_ids
        self.step = int(step)

    def reset(self) -> None:
        self._aux = []
        self._metrics = {}
        self._active_layers = None
        self._do_mask = None

    def selected_layers(self) -> List[int]:
        if self.layer_selection == "fixed":
            return [int(i) for i in (self.layer_indices or [self.n_layers - 1])
                    if 0 <= int(i) < self.n_layers]
        if self.layer_selection == "random":
            return [random.choice(self.stage_end_indices)]
        if self.layer_selection == "stage_end_random":
            return [random.choice(self.stage_end_indices)]
        if self.layer_selection == "last4_random":
            lo = max(0, self.n_layers - 4)
            pool = [i for i in self.stage_end_indices if i >= lo]
            return [random.choice(pool)]
        if self.layer_selection in ("multi", "stage_end_multi"):
            return list(self.stage_end_indices)
        raise ValueError(f"unknown cnn_layer_selection={self.layer_selection}")

    def current_ratio(self) -> float:
        if not self.progressive:
            return float(self.drop_ratio)
        return (float(self.drop_ratio) * (2.0 / math.pi)
                * math.atan(float(self.step) / max(self.velocity, 1e-6)))

    def _domain_ids_for(self, feature: torch.Tensor) -> torch.Tensor | None:
        d = self.domain_ids
        if d is None:
            return None
        d = d.detach().to(device=feature.device, dtype=torch.long).view(-1)
        if d.numel() != feature.shape[0]:
            if d.numel() == 0:
                return None
            d = d.repeat((feature.shape[0] + d.numel() - 1) // d.numel())
            d = d[:feature.shape[0]]
        return d

    @staticmethod
    def _linear_weight(head: nn.Module) -> torch.Tensor:
        return head[-1].weight if isinstance(head, nn.Sequential) else head.weight

    def _domain_aux(self, feature: torch.Tensor, layer_idx: int,
                    did: torch.Tensor) -> None:
        head = self.domain_heads[str(layer_idx)]
        gap = feature.mean(dim=(2, 3))
        grl = self.domain_grl * min(
            1.0, float(self.step) / float(self.grl_warmup_steps))
        logits = head(grad_reverse(gap, grl))
        aux = F.cross_entropy(logits, did)
        self._aux.append(self.domain_lambda * aux)
        self._metrics[f"domain_loss_{layer_idx}"] = float(aux.detach())
        self._metrics[f"domain_acc_{layer_idx}"] = float(
            (logits.argmax(dim=1) == did).float().mean())
        self._metrics[f"feature_norm_{layer_idx}"] = float(
            gap.detach().norm(dim=1).mean())
        self._metrics[f"logit_norm_{layer_idx}"] = float(
            logits.detach().norm(dim=1).mean())
        self._metrics[f"grl_{layer_idx}"] = float(grl)

    def _score(self, feature: torch.Tensor, layer_idx: int,
               did: torch.Tensor | None) -> torch.Tensor:
        b, c = feature.shape[:2]
        if self.mode == "place":
            return torch.rand((b, c), device=feature.device, dtype=feature.dtype)
        gap = feature.detach().mean(dim=(2, 3))
        if self.mode == "dgcd":
            if did is None:
                return torch.rand_like(gap)
            return self.dgcd[str(layer_idx)].score(gap, did)
        if self.mode in ("domaindrop", "domaindrop_norm"):
            if did is None:
                return torch.rand_like(gap)
            weight = self._linear_weight(
                self.domain_heads[str(layer_idx)])[did].detach()
            score = feature.detach() * weight[:, :, None, None]
            smin = score.amin(dim=1, keepdim=True)
            smax = score.amax(dim=1, keepdim=True)
            score = (score - smin) / (smax - smin).clamp_min(self.eps)
            return score.mean(dim=(2, 3))
        raise ValueError(f"unknown cnn_dropout_mode={self.mode}")

    def _apply_dropout(self, feature: torch.Tensor, layer_idx: int,
                       did: torch.Tensor | None) -> torch.Tensor:
        ratio = self.current_ratio()
        if ratio <= 0:
            return feature
        score = self._score(feature, layer_idx, did)
        normalize = "per_sample" if self.mode == "place" else "global"
        mask = wrs_mask(score, drop_ratio=ratio, recover=self.recover,
                        normalize=normalize)
        self._metrics[f"ratio_{layer_idx}"] = float(ratio)
        self._metrics[f"keep_frac_{layer_idx}"] = float(
            (mask > 0).float().mean())
        return feature * mask[:, :, None, None]

    def hook(self, layer_idx: int):
        def _hook(module, inputs, output):
            if self.mode == "none":
                return output
            if self.train_only and not module.training:
                return output
            if self._active_layers is None:
                self._active_layers = self.selected_layers()
                self._do_mask = (random.random() < self.layer_wise_prob)
            did = self._domain_ids_for(output)
            if (self.mode in ("domaindrop", "domaindrop_norm")
                    and str(layer_idx) in self.domain_heads and did is not None):
                self._domain_aux(output, layer_idx, did)
            if (layer_idx in self._active_layers and self._do_mask
                    and self.current_ratio() > 0):
                return self._apply_dropout(output, layer_idx, did)
            return output
        return _hook

    def aux_loss(self) -> torch.Tensor | None:
        if not self._aux:
            return None
        return torch.stack(self._aux).mean()

    def summary(self) -> Dict[str, float]:
        return dict(self._metrics)


def install_cnn_dropout(obs_encoder: nn.Module, **kwargs) -> List[CNNChannelDropoutController]:
    """Attach one controller to every ResNet18ConvFiLM backbone."""
    try:
        from robomimic.models.base_nets import ResNet18ConvFiLM
    except Exception:  # pragma: no cover
        return []
    controllers: List[CNNChannelDropoutController] = []
    for module in obs_encoder.modules():
        if not isinstance(module, ResNet18ConvFiLM):
            continue
        if getattr(module, "_cnn_dropout_installed", False):
            continue
        controller = CNNChannelDropoutController(module._film_layers, **kwargs)
        module.cnn_dropout = controller
        module._cnn_dropout_installed = True
        for idx, layer in enumerate(module._film_layers):
            layer.register_forward_hook(controller.hook(idx))
        controllers.append(controller)
    return controllers


__all__ = ["CNNChannelDropoutController", "install_cnn_dropout"]
