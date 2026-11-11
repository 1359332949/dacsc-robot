"""Residual affine source-mapping at the action interface (CORAL-style).

Theory: for Gaussian features under covariate shift the KL divergence to the
source is minimised by matching BOTH first and second moments (CORAL, Sun &
Saenko 2016; the same family as SSA/PSC's moment matching).  Our previous
versions matched only the mean (debias) or only the per-direction variance
(shrink), which is sub-optimal; here we solve the complete affine map inside a
low-rank residual subspace:

    c   = r P^T                        (top-k residual coordinates)
    c'  = mu_s + Sigma_s^{1/2} Sigma_t^{-1/2} (c - mu_t)
    r'  = r + (c' - c) P               (outside the subspace unchanged)
    w'  = w + r' - r

Source moments: mu_s = P residual_mean, Sigma_s = diag(lambda_s_topk) (the
residual PCA basis diagonalises the source residual covariance).
Target moments: EMA of mean and second moment over frames/tokens.
Numerical safety: covariance shrinkage + eigenvalue clipping of the map to
[0.5, 2] (bounded, reversible); severity gate skips clean / extreme shifts.

The hook uses the verified rule: call 0 = prefix (skip + reset), calls 1..10 =
Euler steps, apply only at call == num_steps.
"""

from __future__ import annotations

import torch


class ResidualAffineCalibration:
    def __init__(
        self,
        attention,
        stats: dict,
        spectrum: dict,
        *,
        k: int = 20,
        num_steps: int = 10,
        action_horizon: int = 12,
        ema: float = 0.9,
        warmup: int = 4,
        shrinkage: float = 0.1,
        map_lo: float = 0.5,
        map_hi: float = 2.0,
        gate_min: float = 1.2,
        gate_max: float = 4.5,
        start_step: int | None = None,
        alpha: float = 1.0,
    ):
        self.attention = attention
        self.original = attention.attend
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.basis = torch.tensor(stats["basis"], dtype=torch.float32)
        basis_topk = torch.tensor(spectrum["residual_basis_topk"], dtype=torch.float32)
        eigvals = torch.tensor(spectrum["residual_eigvals_topk"], dtype=torch.float32)
        k = min(int(k), basis_topk.shape[0])
        self.k = k
        self.residual_basis = basis_topk[:k].contiguous()          # [k, D]
        source_mean = torch.tensor(spectrum["residual_mean"], dtype=torch.float32)
        self.mu_s = self.residual_basis @ source_mean               # [k]
        self.cov_s = torch.diag(eigvals[:k].clamp_min(1e-6))        # [k, k]
        self.tau_s = float(spectrum["tau_residual"])
        self.num_steps = int(num_steps)
        self.action_horizon = int(action_horizon)
        self.ema = float(ema)
        self.warmup = int(warmup)
        self.shrinkage = float(shrinkage)
        self.map_lo = float(map_lo)
        self.map_hi = float(map_hi)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.start_step = int(start_step) if start_step is not None else int(num_steps)
        self.alpha = float(alpha)
        self.step = 0
        self.steps = 0
        self.mu_t: torch.Tensor | None = None
        self.second_t: torch.Tensor | None = None
        self.tau_t: float | None = None
        self.mode = "warmup"
        self.mode_counts = {"affine": 0, "skip": 0, "warmup": 0}
        attention.attend = self._attend  # type: ignore[assignment]

    def _map(self, device: torch.device) -> torch.Tensor:
        cov_s = self.cov_s.to(device).float()
        mu = self.mu_t.to(device).float()
        second = self.second_t.to(device).float()
        cov_t = second - torch.outer(mu, mu)
        # shrink the estimated target covariance towards the SOURCE covariance
        # (a proper prior): this keeps the map exactly identity when the target
        # moments already equal the source, whereas shrinking towards an
        # isotropic matrix biases the map even under no shift.
        cov_t = (1 - self.shrinkage) * cov_t + self.shrinkage * cov_s
        values, vectors = torch.linalg.eigh(cov_s + 1e-6 * torch.eye(self.k, device=device))
        sqrt_s = (vectors * values.clamp_min(1e-6).sqrt()) @ vectors.T
        values_t, vectors_t = torch.linalg.eigh(cov_t + 1e-6 * torch.eye(self.k, device=device))
        invsqrt_t = (vectors_t * values_t.clamp_min(1e-6).rsqrt()) @ vectors_t.T
        mapping = (sqrt_s @ invsqrt_t).float()
        mvals, mvec = torch.linalg.eigh(mapping)
        mapping = (mvec * mvals.clamp(self.map_lo, self.map_hi)) @ mvec.T
        return mapping

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        if output.shape[1] != self.action_horizon:
            self.step = 0
            return output
        self.step += 1
        device = output.device
        mean = self.mean.to(device)
        basis = self.basis.to(device)
        residual_basis = self.residual_basis.to(device)
        centred = output - mean
        u = centred @ basis.T
        residual = centred - u @ basis
        residual32 = residual.float()
        coords = (residual32 @ residual_basis.T).reshape(-1, self.k)   # [N, k]
        batch_mean = coords.mean(dim=0).detach()
        batch_second = (coords.T @ coords / coords.shape[0]).detach()
        energy = float(residual.pow(2).sum(dim=-1).mean())
        self.mu_t = batch_mean if self.mu_t is None else self.ema * self.mu_t + (1 - self.ema) * batch_mean
        self.second_t = batch_second if self.second_t is None else (
            self.ema * self.second_t + (1 - self.ema) * batch_second
        )
        self.tau_t = energy if self.tau_t is None else self.ema * self.tau_t + (1 - self.ema) * energy
        self.steps += 1
        if self.steps <= self.warmup:
            self.mode = "warmup"
            self.mode_counts["warmup"] += 1
            return output
        if self.step < self.start_step:
            self.mode = "warmup"
            self.mode_counts["warmup"] += 1
            return output
        ratio = self.tau_t / max(self.tau_s, 1e-12)
        if ratio < self.gate_min or ratio > self.gate_max:
            self.mode = "skip"
            self.mode_counts["skip"] += 1
            return output
        mapping = self._map(device)
        coords2 = (residual32 @ residual_basis.T).reshape(-1, self.k)
        mapped = self.mu_s.to(device) + (coords2 - self.mu_t) @ mapping.T
        mapped = coords2 + self.alpha * (mapped - coords2)
        delta = ((mapped - coords2) @ residual_basis).to(output.dtype)
        self.mode = "affine"
        self.mode_counts["affine"] += 1
        return output + delta.reshape(residual.shape)

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
