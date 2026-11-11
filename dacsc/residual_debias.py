"""Residual mean-shift correction at the action interface (source mapping).

Diagnosis (write_spectrum_probe):
    clean : tau_res=4.18, residual mean energy=1.81 (43% of total)
    camera: tau_res=14.88, residual mean energy=9.65 (65% of total)
    noise : excess is variance-dominated (mean share much smaller)
=> camera-like shifts are a consistent bias; correct the mean, not the variance.

    r = (w - mu_s) - V^T V (w - mu_s)
    r' = r - alpha * (mean_t(r) - mean_s(r))
    w' = w + (r' - r)
with mean_s stored in the source spectrum and mean_t tracked with an EMA.
Same hook rule as the other modules (prefix skipped, only last Euler step).
"""

from __future__ import annotations

import torch


class ResidualDebias:
    def __init__(
        self,
        attention,
        stats: dict,
        spectrum: dict,
        *,
        num_steps: int = 10,
        action_horizon: int = 12,
        alpha: float = 1.0,
        mean_ratio_min: float = 4.5,
        ema: float = 0.9,
        warmup: int = 4,
        ratio_min: float = 1.15,
    ):
        self.attention = attention
        self.original = attention.attend
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.basis = torch.tensor(stats["basis"], dtype=torch.float32)
        self.source_mean = torch.tensor(spectrum["residual_mean"], dtype=torch.float32)
        self.num_steps = int(num_steps)
        self.action_horizon = int(action_horizon)
        self.alpha = float(alpha)
        self.mean_ratio_min = float(mean_ratio_min)
        self.ratio_min = float(ratio_min)
        self.source_mean_energy = float(torch.sum(self.source_mean ** 2))
        self.target_mean_energy: float | None = None
        self.ema = float(ema)
        self.warmup = int(warmup)
        self.step = 0
        self.steps = 0
        self.target_mean: torch.Tensor | None = None
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        if output.shape[1] != self.action_horizon:
            self.step = 0
            return output
        self.step += 1
        if self.step != self.num_steps:
            return output
        device = output.device
        mean = self.mean.to(device)
        basis = self.basis.to(device)
        source_mean = self.source_mean.to(device)
        centred = output - mean
        u = centred @ basis.T
        residual = centred - u @ basis
        batch_mean = residual.reshape(-1, residual.shape[-1]).mean(dim=0).detach()
        self.target_mean = batch_mean if self.target_mean is None else self.ema * self.target_mean + (1 - self.ema) * batch_mean
        batch_energy = float(torch.sum(batch_mean ** 2))
        self.target_mean_energy = batch_energy if self.target_mean_energy is None else (
            self.ema * self.target_mean_energy + (1 - self.ema) * batch_energy
        )
        self.steps += 1
        if self.steps <= self.warmup:
            return output
        # only correct a mean shift that is clearly larger than the source one
        # (clean data has a stable non-zero residual mean too; correcting it
        #  blindly cost 10 points on clean -- measured, so we gate it).
        if self.source_mean_energy > 0 and self.target_mean_energy / self.source_mean_energy < self.ratio_min:
            return output
        correction = -self.alpha * (self.target_mean - source_mean)
        return output + correction

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
