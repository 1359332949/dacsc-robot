"""Residual spectral suppression at the action interface (our method, corrected).

Measurement (write_spectrum_probe, 240 frames per condition, fixed subspaces):
    clean : energy_A=18.98  energy_N=25.28  energy_residual=4.18
    noise4: energy_A=26.74  energy_N=32.71  energy_residual=19.56
    noise5: energy_A=20.48  energy_N=24.67  energy_residual=21.16
i.e. the domain shift lands mainly in the *residual* (4.2 -> ~20), while the
nuisance support spectrum barely changes (sigma ratio ~0.8-1.0).  Therefore the
correct intervention of our original idea (source-subspace + sensitivity +
spectrum + soft suppression) is:

    r  = (w - mu_s) - V^T V (w - mu_s)        residual (noise-dominated)
    gamma = clip(sqrt(tau_s / tau_t), gamma_min, 1.0)
    w' = w + (gamma - 1) * r                  soft shrink, reversible, bounded

Action-sensitive support coordinates are untouched; the shrink factor is
computed from the source residual spectrum (PSC/SSA residual term, but as a soft
scale rather than a hard projection).
"""

from __future__ import annotations

import torch


class ResidualShrink:
    def __init__(
        self,
        attention,
        stats: dict,
        *,
        num_steps: int = 10,
        action_horizon: int = 12,
        tau_s: float | None = None,
        gamma_min: float = 0.3,
        ema: float = 0.9,
        warmup: int = 4,
    ):
        self.attention = attention
        self.original = attention.attend
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.basis = torch.tensor(stats["basis"], dtype=torch.float32)
        self.num_steps = int(num_steps)
        self.action_horizon = int(action_horizon)
        # tau_s is the source residual spectrum measured by write_spectrum_probe;
        # it must be passed explicitly (no hidden magic constant).
        if tau_s is None:
            tau_s = stats.get("tau_residual")
        if tau_s is None:
            raise ValueError("tau_s must be provided (source residual energy)")
        self.tau_s = float(tau_s)
        self.gamma_min = float(gamma_min)
        self.ema = float(ema)
        self.warmup = int(warmup)
        self.step = 0
        self.steps = 0
        self.tau_t: float | None = None
        self.gamma = 1.0
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        # the prefix-encoding call runs over representation tokens (T != horizon);
        # it also marks the start of a new predict, so reset the Euler counter
        if output.shape[1] != self.action_horizon:
            self.step = 0
            return output
        self.step += 1
        if self.step != self.num_steps:
            return output
        device = output.device
        mean = self.mean.to(device)
        basis = self.basis.to(device)
        centred = output - mean
        u = centred @ basis.T
        residual = centred - u @ basis
        energy = float(residual.pow(2).sum(dim=-1).mean())
        self.tau_t = energy if self.tau_t is None else self.ema * self.tau_t + (1 - self.ema) * energy
        self.steps += 1
        if self.steps <= self.warmup:
            return output
        self.gamma = min(1.0, max(self.gamma_min, (self.tau_s / max(self.tau_t, 1e-8)) ** 0.5))
        return output + (self.gamma - 1.0) * residual

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
