"""Appearance-consistency regularisation for the CoRP flow policy.

Why this exists
---------------
V9 showed the action offset caused by an appearance shift cannot be predicted
from test-time observables, so inference-time compensation is not deployable.
The remaining internal route is to change what the model depends on during
training: make the predicted flow velocity invariant to an appearance change of
the SAME state and SAME flow sample.

Implementation provenance (mirrors released/verified code, no invented formula)
----------------------------------------------------------------------------
* paired forward with identical flow noise: `channel_training_loss` in
  scripts/train_corp_robocasa.py (lines 367-408) uses
  `duplicate_training_batch` + `paired_flow_sample` + `training_loss_with_flow`
  and then `consistency = F.mse_loss(velocity_a, velocity_b)`.
  This module reproduces that structure with the second half's IMAGES replaced
  by an appearance view instead of the second half's channels dropped.
* `training_loss_with_flow(..., eps=, tau=)` already exists in
  dacsc/corp_robocasa.py:323 and documents "paired forwards can reuse exactly
  the same flow sample" -- that is the hook used here.
* consistency term: Mean Teacher (Tarvainen & Valpola, NeurIPS 2017) MSE
  consistency with sigmoid ramp-up; the paired-variant objective for VLA is
  PAIR-VLA (arXiv:2605.13105, task-preserving pairs keep the action); the
  weight default 1.5 follows DomainDrop's KL_Loss_weight
  (loss/KL_Loss.py: symmetric divergence, weight 1.5). For continuous
  velocities KL does not apply, so MSE is used -- the same substitution the
  CoRP training code already makes for its channel-consistency term.
* photometric view: same ops as dacsc/corp_images (brightness/contrast/
  saturation from torchvision), applied in the unnormalised space then
  renormalised, exactly like prepare_image's order resize -> shift -> normalise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dacsc.corp_images import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class AppearanceConsistencyConfig:
    weight: float = 1.5            # DomainDrop KL_Loss weight default
    brightness: float = 0.4        # +- relative strength of the random view
    contrast: float = 0.4
    saturation: float = 0.4
    ramp: str = "sigmoid"          # "sigmoid" (Mean Teacher) | "constant"
    ramp_steps: int = 500
    stop_grad: bool = True         # one side detached (Mean Teacher)
    duplicate_only: bool = False   # ablation: doubled batch WITHOUT consistency

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("appearance consistency weight must be nonnegative")
        for name in ("brightness", "contrast", "saturation"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} strength must be in [0, 1)")
        if self.ramp not in ("sigmoid", "constant"):
            raise ValueError("ramp must be 'sigmoid' or 'constant'")
        if self.ramp_steps <= 0:
            raise ValueError("ramp_steps must be positive")

    def to_dict(self) -> dict:
        return {
            "weight": self.weight,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "saturation": self.saturation,
            "ramp": self.ramp,
            "ramp_steps": self.ramp_steps,
            "stop_grad": self.stop_grad,
            "duplicate_only": self.duplicate_only,
        }


def ramp_factor(config: AppearanceConsistencyConfig, step: int) -> float:
    """Mean Teacher sigmoid ramp-up: w(t) = exp(-5 (1 - t/T)^2)."""
    if config.ramp == "constant":
        return 1.0
    progress = min(max(step / float(config.ramp_steps), 0.0), 1.0)
    return math.exp(-5.0 * (1.0 - progress) ** 2)


def photometric_view(
    images: dict[str, torch.Tensor],
    config: AppearanceConsistencyConfig,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Random appearance view of the same observations (normalised space in/out)."""
    from torchvision.transforms import functional as TVF

    out: dict[str, torch.Tensor] = {}
    # one photometric transform per scene, shared by every camera view: the
    # appearance change is a property of the scene, not of one camera.
    brightness = 1.0 + (torch.rand(1, generator=generator).item() * 2.0 - 1.0) * config.brightness
    contrast = 1.0 + (torch.rand(1, generator=generator).item() * 2.0 - 1.0) * config.contrast
    saturation = 1.0 + (torch.rand(1, generator=generator).item() * 2.0 - 1.0) * config.saturation
    for name, value in images.items():
        mean = IMAGENET_MEAN.to(device=value.device, dtype=value.dtype).view(1, 3, 1, 1)
        std = IMAGENET_STD.to(device=value.device, dtype=value.dtype).view(1, 3, 1, 1)
        rgb = (value * std + mean).clamp(0.0, 1.0)
        rgb = TVF.adjust_brightness(rgb, brightness)
        rgb = TVF.adjust_contrast(rgb, contrast)
        rgb = TVF.adjust_saturation(rgb, saturation)
        out[name] = ((rgb - mean) / std).to(dtype=value.dtype)
    return out


def duplicate_training_batch(batch: dict) -> dict:
    """Mirror of scripts/train_corp_robocasa.py:333 (two forwards, one update)."""
    out = {
        "images": {
            view: torch.cat([image, image], dim=0)
            for view, image in batch["images"].items()
        },
        "state": torch.cat([batch["state"], batch["state"]], dim=0),
        "instruction": list(batch["instruction"]) * 2,
        "action": torch.cat([batch["action"], batch["action"]], dim=0),
        "action_mask": torch.cat([batch["action_mask"], batch["action_mask"]], dim=0),
        "style_id": torch.cat([batch["style_id"], batch["style_id"]], dim=0),
    }
    if "task_id" in batch:
        out["task_id"] = torch.cat([batch["task_id"], batch["task_id"]], dim=0)
    return out


def paired_flow_sample(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror of scripts/train_corp_robocasa.py:353 (identical eps/tau per pair)."""
    if action.ndim < 1 or action.shape[0] % 2:
        raise ValueError("paired action batch must contain two equal halves")
    half_batch = action.shape[0] // 2
    eps_half = torch.randn_like(action[:half_batch])
    tau_half = (
        torch.distributions.Beta(1.0, 1.5)
        .sample((half_batch,))
        .to(device=action.device, dtype=action.dtype)
    )
    return torch.cat([eps_half, eps_half], dim=0), torch.cat([tau_half, tau_half], dim=0)


def appearance_consistency_loss(
    model,
    batch: dict,
    config: AppearanceConsistencyConfig,
    *,
    step: int,
    generator: torch.Generator,
    task_labels: torch.Tensor | None,
) -> tuple[torch.Tensor, dict]:
    """Flow loss on (clean, appearance-view) pair + MSE velocity consistency.

    Returns (total_loss, losses, diagnostics). The total already includes
    config.weight * ramp(step) * consistency, so the caller adds it directly.
    """
    if config.weight == 0:
        raise ValueError("zero-weight consistency should be handled by the caller")
    doubled = duplicate_training_batch(batch)
    view = photometric_view(batch["images"], config, generator)
    for name in doubled["images"]:
        doubled["images"][name] = torch.cat(
            [batch["images"][name], view[name]], dim=0
        )
    eps, tau = paired_flow_sample(doubled["action"])
    losses, velocity = model.training_loss_with_flow(
        doubled["images"],
        doubled["state"],
        doubled["instruction"],
        doubled["action"],
        doubled["action_mask"],
        task_labels=doubled.get("task_id") if task_labels is not None else None,
        eps=eps,
        tau=tau,
    )
    half = batch["action"].shape[0]
    velocity_a = velocity[:half]
    velocity_b = velocity[half:]
    if config.stop_grad:
        velocity_b = velocity_b.detach()
    if config.duplicate_only:
        consistency = velocity_a.new_zeros(())
    else:
        consistency = torch.nn.functional.mse_loss(velocity_a.float(), velocity_b.float())
    total = losses.total + config.weight * ramp_factor(config, step) * consistency
    diagnostics = {
        "appearance_consistency": float(consistency.detach()),
        "appearance_ramp": ramp_factor(config, step),
        "appearance_weight": config.weight,
    }
    return total, losses, diagnostics
