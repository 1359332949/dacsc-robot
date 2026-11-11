"""Soft, action-masked calibration of the cross-attention write.

Motivation (measured on the official CoRP checkpoint): the write produced by
`action_generator.blocks[0].attention.attend` is the interface where visual
information enters the action tokens, and the source-support projection applied
there deletes a large fraction of the action-sensitive energy. Instead of a
hard projection we apply a reversible per-dimension affine calibration:

    z' = mu_s + (z - mu_s) * alpha_d

with alpha_d = clip((sigma_s,d / sigma_t,d) ** expo, lo, hi) for nuisance
dimensions, and alpha_d = 1 for action-sensitive dimensions (mask). Source
statistics come from clean writes (`libero_psc_stats.json`); target statistics
are tracked online with an EMA. No gradients, no weight updates.
"""

from __future__ import annotations

import math

import torch


def source_diagonal(stats: dict) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    basis = torch.tensor(stats["basis"], dtype=torch.float32)
    lambdas = torch.tensor(stats["lambdas"] if "lambdas" in stats else stats["sigma_s"], dtype=torch.float32)
    tau = float(stats.get("tau", 0.0))
    if "lambdas" in stats:
        variance = (basis.T ** 2) @ lambdas + tau
    else:  # legacy stats store coordinate stds
        variance = (basis.T ** 2) @ (lambdas ** 2) + tau
    return mean, variance.clamp_min(1e-8)


class WriteCalibration:
    def __init__(
        self,
        attention,
        stats: dict,
        *,
        mode: str = "support",
        mask_indices: list[int] | None = None,
        expo: float = 1.0,
        # mode is documented above
        lo: float = 0.5,
        hi: float = 2.0,
        ema: float = 0.9,
        min_steps: int = 4,
        num_steps: int = 10,
        last_step_only: bool = True,
    ):
        self.attention = attention
        self.original = attention.attend
        self.mean_s, self.var_s = source_diagonal(stats)
        self.mode = mode
        self.basis = torch.tensor(stats["basis"], dtype=torch.float32)
        coords = stats.get("sigma_s")
        if coords is None:
            coords = [1.0] * self.basis.shape[0]
        self.coords_std = torch.tensor(coords, dtype=torch.float32).clamp_min(1e-6)
        self.mask = torch.zeros_like(self.mean_s, dtype=torch.bool)
        if mask_indices:
            self.mask[torch.tensor(mask_indices, dtype=torch.long)] = True
        self.expo = expo
        self.lo = lo
        self.hi = hi
        self.ema = ema
        self.min_steps = min_steps
        self.num_steps = max(1, int(num_steps))
        self.last_step_only = last_step_only
        self.steps = 0
        self.call = 0
        self.mean_t: torch.Tensor | None = None
        self.var_t: torch.Tensor | None = None
        self.coord_mean_t: torch.Tensor | None = None
        self.coord_var_t: torch.Tensor | None = None
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        device = output.device
        self.call += 1
        is_last = (self.call % self.num_steps) == 0
        mean_s = self.mean_s.to(device)
        var_s = self.var_s.to(device)
        mask = self.mask.to(device)
        if self.mode == "support":
            basis = self.basis.to(device)
            mean_s = mean_s.to(device)
            coords = (output - mean_s) @ basis.T  # [B,T,K]
            batch_coord_mean = coords.reshape(-1, self.basis.shape[0]).mean(dim=0).detach()
            batch_coord_var = coords.reshape(-1, self.basis.shape[0]).var(dim=0).detach()
            if self.coord_mean_t is None:
                self.coord_mean_t = batch_coord_mean
                self.coord_var_t = batch_coord_var
            else:
                self.coord_mean_t = self.ema * self.coord_mean_t + (1 - self.ema) * batch_coord_mean
                self.coord_var_t = self.ema * self.coord_var_t + (1 - self.ema) * batch_coord_var
            self.steps += 1
            if self.steps < self.min_steps:
                return output
            if self.last_step_only and not is_last:
                return output
            sigma_t = self.coord_var_t.clamp_min(1e-8).sqrt()
            scale = (self.coords_std.to(device) / sigma_t).clamp(self.lo, self.hi).pow(self.expo)
            correction = ((scale - 1.0).unsqueeze(0).unsqueeze(0) * coords) @ basis
            return output + correction
        flat = output.reshape(-1, output.shape[-1])
        batch_mean = flat.mean(dim=0).detach()
        batch_var = flat.var(dim=0).detach()
        if self.mean_t is None:
            self.mean_t = batch_mean
            self.var_t = batch_var
        else:
            self.mean_t = self.ema * self.mean_t + (1 - self.ema) * batch_mean
            self.var_t = self.ema * self.var_t + (1 - self.ema) * batch_var
        self.steps += 1
        if self.steps < self.min_steps:
            return output
        if self.last_step_only and not is_last:
            return output
        sigma_s = var_s.sqrt()
        sigma_t = self.var_t.clamp_min(1e-8).sqrt()
        scale = (sigma_s / sigma_t).clamp(self.lo, self.hi).pow(self.expo)
        scale = torch.where(mask, torch.ones_like(scale), scale)
        centred = output - mean_s
        return mean_s + centred * scale

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
