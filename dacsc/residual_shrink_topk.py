"""Selective residual spectral suppression (top-k residual directions only).

Measured facts (write_spectrum_probe, 240 frames per condition):
  clean residual energy 4.18, L4 19.56, L5 21.16; residual energy concentrated
  in the leading ~20 residual eigendirections (top20 = 88-94%).
Therefore we suppress only the leading k residual directions:

    u = V (w - mu_s)                      support coordinates
    r = (w - mu_s) - V^T u                residual (noise-dominated)
    c = r P^T                             coords on residual basis P (k x D)
    gamma_i = clip(sqrt(lambda_s_i / var_t_i), gamma_min, 1)   (per direction)
    r_top' = ((gamma - 1) * c) P          shrink excess energy only
    w' = w + r_top'                       rest of the residual untouched

Action-sensitive directions live in the support subspace, which is never touched.
Hook rule (verified): call 0 = prefix (129 tokens, skip and reset), calls 1..10 =
Euler steps, apply only at call == num_steps; reset per predict via token count.
"""

from __future__ import annotations

import torch


class TopKResidualShrink:
    def __init__(
        self,
        attention,
        stats: dict,
        spectrum: dict,
        *,
        k: int = 20,
        num_steps: int = 10,
        action_horizon: int = 12,
        gamma_min: float = 0.5,
        ratio_min: float = 1.2,
        ratio_max: float = 2.0,
        ema: float = 0.9,
        warmup: int = 4,
    ):
        self.attention = attention
        self.original = attention.attend
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.basis = torch.tensor(stats["basis"], dtype=torch.float32)
        basis_topk = torch.tensor(spectrum["residual_basis_topk"], dtype=torch.float32)
        eigvals_topk = torch.tensor(spectrum["residual_eigvals_topk"], dtype=torch.float32)
        k = min(int(k), basis_topk.shape[0])
        self.residual_basis = basis_topk[:k].contiguous()      # [k, D]
        self.lambda_s = eigvals_topk[:k].clamp_min(1e-8)       # source variances
        self.k = k
        self.num_steps = int(num_steps)
        self.action_horizon = int(action_horizon)
        self.gamma_min = float(gamma_min)
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.tau_s = float(spectrum.get("tau_residual", 0.0))
        self.tau_t: float | None = None
        self.ema = float(ema)
        self.warmup = int(warmup)
        self.step = 0
        self.steps = 0
        self.var_t: torch.Tensor | None = None
        self.gamma = torch.ones(k)
        self.gate_off = False
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        if output.shape[1] != self.action_horizon:   # prefix encoding: never touch
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
        coords = residual @ residual_basis.T                  # [B, T, k]
        batch_var = coords.reshape(-1, self.k).pow(2).mean(dim=0).detach()
        self.var_t = batch_var if self.var_t is None else self.ema * self.var_t + (1 - self.ema) * batch_var
        # shift-severity gate: suppress only in the moderate-excess band measured
        # empirically (L3 ratio ~3.3 -> helpful; L4/L5 ratio ~5 -> harmful).
        if self.tau_t is None:
            self.tau_t = float(residual.pow(2).sum(dim=-1).mean())
        else:
            self.tau_t = self.ema * self.tau_t + (1 - self.ema) * float(residual.pow(2).sum(dim=-1).mean())
        self.steps += 1
        if self.steps <= self.warmup:
            return output
        if self.tau_s > 0:
            ratio = self.tau_t / self.tau_s
            if ratio < self.ratio_min or ratio > self.ratio_max:
                self.gate_off = True
                return output
        self.gate_off = False
        gamma = (self.lambda_s.to(device) / self.var_t.clamp_min(1e-8)).sqrt().clamp(self.gamma_min, 1.0)
        self.gamma = gamma.detach()
        correction = ((gamma - 1.0) * coords) @ residual_basis
        return output + correction

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
