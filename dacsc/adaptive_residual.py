"""Adaptive residual source-mapping at the action interface (our method, unified).

The correction is chosen from the *measured residual structure* rather than from
the perturbation label:

    r = (w - mu_s) - V^T V (w - mu_s)
    mean-shift ratio  r_mean = E[||mean_t(r)||^2] / ||mean_s(r)||^2
    energy ratio      r_eng  = E[||r||^2] / tau_s

    r_mean >= 4.5                      -> DEBIAS  (camera-like, consistent bias)
    1.2 <= r_eng <= 4.5                -> SHRINK  (noise-like, variance excess)
    otherwise                          -> SKIP    (clean / too severe)

Thresholds come from measured distributions: clean mean-shift ratio <= 3.66,
camera >= 6.03; clean/noise energy ratio bands as measured before.
All operations are reversible, bounded and applied only at the last Euler step;
the action-sensitive support subspace is never touched.
"""

from __future__ import annotations

import torch


class AdaptiveResidualMapping:
    def __init__(
        self,
        attention,
        stats: dict,
        spectrum: dict,
        *,
        num_steps: int = 10,
        action_horizon: int = 12,
        k: int = 10,
        gamma_min: float = 0.5,
        mean_ratio_min: float = 4.5,
        energy_ratio_min: float = 1.2,
        energy_ratio_max: float = 4.5,
        ema: float = 0.9,
        warmup: int = 4,
        lock_after: int = 10,
    ):
        self.attention = attention
        self.original = attention.attend
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.basis = torch.tensor(stats["basis"], dtype=torch.float32)
        basis_topk = torch.tensor(spectrum["residual_basis_topk"], dtype=torch.float32)
        eigvals = torch.tensor(spectrum["residual_eigvals_topk"], dtype=torch.float32)
        self.k = min(int(k), basis_topk.shape[0])
        self.residual_basis = basis_topk[: self.k].contiguous()
        self.lambda_s = eigvals[: self.k].clamp_min(1e-8)
        self.source_mean = torch.tensor(spectrum["residual_mean"], dtype=torch.float32)
        self.tau_s = float(spectrum["tau_residual"])
        self.source_mean_energy = float(torch.sum(self.source_mean ** 2))
        self.num_steps = int(num_steps)
        self.action_horizon = int(action_horizon)
        self.gamma_min = float(gamma_min)
        self.mean_ratio_min = float(mean_ratio_min)
        self.energy_ratio_min = float(energy_ratio_min)
        self.energy_ratio_max = float(energy_ratio_max)
        self.ema = float(ema)
        self.warmup = int(warmup)
        self.lock_after = int(lock_after)
        self.mode_locked: str | None = None
        self.step = 0
        self.steps = 0
        self.target_mean: torch.Tensor | None = None
        self.var_t: torch.Tensor | None = None
        self.tau_t: float | None = None
        self.mode = "skip"
        self.mode_counts = {"debias": 0, "shrink": 0, "skip": 0, "warmup": 0}
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
        residual_basis = self.residual_basis.to(device)
        centred = output - mean
        u = centred @ basis.T
        residual = centred - u @ basis
        flat = residual.reshape(-1, residual.shape[-1])
        batch_mean = flat.mean(dim=0).detach()
        batch_var = (flat @ residual_basis.T).pow(2).mean(dim=0).detach()
        energy = float(flat.pow(2).sum(dim=-1).mean())
        self.target_mean = batch_mean if self.target_mean is None else self.ema * self.target_mean + (1 - self.ema) * batch_mean
        self.var_t = batch_var if self.var_t is None else self.ema * self.var_t + (1 - self.ema) * batch_var
        self.tau_t = energy if self.tau_t is None else self.ema * self.tau_t + (1 - self.ema) * energy
        self.steps += 1
        if self.steps <= self.warmup:
            self.mode = "warmup"
            self.mode_counts["warmup"] += 1
            return output
        ratio_mean = float(torch.sum(self.target_mean ** 2)) / max(self.source_mean_energy, 1e-12)
        ratio_energy = self.tau_t / max(self.tau_s, 1e-12)
        # decide the mode ONCE from accumulated evidence and keep it; per-frame
        # switching jitters (measured: L3 run selected debias 27x and shrink 60x).
        if self.mode_locked is None and self.steps >= self.lock_after:
            if ratio_mean >= self.mean_ratio_min:
                self.mode_locked = "debias"
            elif self.energy_ratio_min <= ratio_energy <= self.energy_ratio_max:
                self.mode_locked = "shrink"
            else:
                self.mode_locked = "skip"
        if self.mode_locked == "debias":
            self.mode = "debias"
            self.mode_counts["debias"] += 1
            correction = -(self.target_mean - self.source_mean.to(device))
            return output + correction
        if self.mode_locked == "shrink":
            self.mode = "shrink"
            self.mode_counts["shrink"] += 1
            coords = flat @ residual_basis.T
            gamma = (self.lambda_s.to(device) / self.var_t.clamp_min(1e-8)).sqrt().clamp(self.gamma_min, 1.0)
            shrink = ((gamma - 1.0) * coords) @ residual_basis
            return output + shrink.reshape(residual.shape)
        self.mode = "skip"
        if self.mode_locked is None:
            self.mode_counts["warmup"] += 1
        else:
            self.mode_counts["skip"] += 1
        return output

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
