"""Interface Subspace Anchoring (ISA) -- our test-time adaptation method.

Definition (fixed):
  site       : cross-attention write of block 0 (action tokens only; the
               prefix-encoding call over representation tokens is skipped).
  transform  : bounded log-scaling inside the source support subspace
                   u  = V (w - mu_s)                (K coordinates)
                   u' = exp(beta) * u               (|beta| <= beta_max)
                   w' = w + V^T (u' - u)            (additive, reversible)
  objective  : L = L_action + lambda_src * L_source + lambda_reg * ||beta||^2
      L_action : masked MSE between the regenerated chunk head and the
                 previous chunk's unexecuted tail (leftover consistency).
      L_source : symmetric KL between N(0, diag(Sigma_s)) and the empirical
                 distribution of the last-Euler-step write coordinates
                 (the source-anchoring term; same family as SSA/PSC but used
                 as a regulariser instead of the sole objective).

References for each piece (no invented conventions):
  * bounded multiplicative scaling / additive reversible form: LoRA-style
    low-rank adaptation (Hu et al., 2021) restricted to a fixed orthonormal
    subspace;
  * symmetric KL formula: SSA official code utils/loss.py
    (diagonal_gaussian_kl_loss, two-directional sum);
  * support statistics: SSA/PSC (PCA support + per-coordinate variances);
  * leftover consistency: SALT (2610.07946) leftover supervision; the target
    is always detached and comes from the previous replan's chunk tail.
"""

from __future__ import annotations

import math

import torch


def normalized_anchor_loss(mean: torch.Tensor, var: torch.Tensor, target_var: torch.Tensor) -> torch.Tensor:
    """Scale-free anchor distance used as a regulariser.

    mean(m^2 / lambda) + mean((v / lambda - 1)^2).  The SSA symmetric-KL form is
    unnormalised (its magnitude depends on the target variance units); for a
    regulariser we use this normalised variant and document the difference.
    """
    ratio_m = (mean.square() / target_var).mean()
    ratio_v = ((var / target_var - 1.0).square()).mean()
    return ratio_m + ratio_v


def symmetric_gaussian_kl(mean: torch.Tensor, var: torch.Tensor, target_var: torch.Tensor) -> torch.Tensor:
    """Two-directional KL between N(mean, var) and N(0, target_var), per dim.

    Mirrors SSA utils/loss.py::diagonal_gaussian_kl_loss (v2=target, v1=pred):
      forward  : 0.5 * (log(v2/v1) + (v1 + (m2-m1)^2)/v2 - 1)
      backward : 0.5 * (log(v1/v2) + (v2 + (m2-m1)^2)/v1 - 1)
    with eps = 1e-8 as in the official code.
    """
    eps = 1e-8
    m1, v1, v2 = mean, var, target_var
    forward = 0.5 * (torch.log(v2 + eps) - torch.log(v1 + eps) + (v1 + m1.square()) / (v2 + eps) - 1.0)
    backward = 0.5 * (torch.log(v1 + eps) - torch.log(v2 + eps) + (v2 + m1.square()) / (v1 + eps) - 1.0)
    return (forward + backward).sum()


class InterfaceAnchoring:
    def __init__(
        self,
        attention,
        stats: dict,
        *,
        num_steps: int = 10,
        beta_max: float = 0.3,
        lr: float = 0.05,
        steps: int = 5,
        lambda_src: float = 0.1,
        lambda_reg: float = 1e-3,
    ):
        self.attention = attention
        self.original = attention.attend
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.basis = torch.tensor(stats["basis"], dtype=torch.float32)          # [K, D]
        support_cov = torch.tensor(stats["support_cov"], dtype=torch.float32)   # [K, K]
        target_var = torch.diag(support_cov).clamp_min(1e-6)
        # dimensionless anchor: normalise by the mean target variance and floor
        # tiny trailing components (they would otherwise dominate the ratio)
        self.target_var = (target_var / target_var.mean()).clamp_min(1e-2)
        self.num_steps = int(num_steps)
        self.beta_max = float(beta_max)
        self.steps = int(steps)
        self.lr = float(lr)
        self.lambda_src = float(lambda_src)
        self.lambda_reg = float(lambda_reg)
        rank = self.basis.shape[0]
        device = next(attention.parameters()).device
        self.beta = torch.zeros(rank, device=device, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.beta], lr=self.lr)
        self.call = 0
        self.active = False
        self.source_loss: torch.Tensor | None = None
        attention.attend = self._attend  # type: ignore[assignment]

    def _transform(self, output: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(output.device)
        basis = self.basis.to(output.device)
        u = (output - mean) @ basis.T                      # [B, T, K]
        scaled = u * torch.exp(self.beta).to(output.device)
        return output + (scaled - u) @ basis

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        index = self.call
        self.call += 1
        if index == 0:
            return output
        if not self.active:
            return output
        output = self._transform(output)
        if index == self.num_steps:
            basis = self.basis.to(output.device)
            mean = self.mean.to(output.device)
            u = (output - mean) @ basis.T
            mean_u = u.reshape(-1, u.shape[-1]).mean(dim=0)
            var_u = u.reshape(-1, u.shape[-1]).var(dim=0, unbiased=False)
            self.source_loss = normalized_anchor_loss(mean_u, var_u, self.target_var.to(output.device))
        return output

    def reset(self):
        self.call = 0
        self.source_loss = None

    def adapt(self, loss_action: torch.Tensor) -> list[float]:
        """Run `steps` optimisation steps on beta; returns the loss trace."""
        trace = []
        for _ in range(self.steps):
            self.active = True
            loss = loss_action
            if self.source_loss is not None:
                loss = loss + self.lambda_src * self.source_loss
            loss = loss + self.lambda_reg * self.beta.square().sum()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.beta.clamp_(-self.beta_max, self.beta_max)
            trace.append(float(loss.detach()))
            loss_action = loss_action.detach()
        return trace

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
