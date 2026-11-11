"""Full Predictive Spectral Calibration (PSC) as test-time training.

Implements the two losses of PSC (arXiv:2603.09338):
  * support-space spectral matching over the K^2 probe bank (Eq. 16)
  * residual-space spectral matching with the isotropic floor tau (Eq. 23)
Only affine parameters of normalization layers in the visual feature extractor
are updated; the action head stays frozen (Eq. 25).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PSCTTA:
    def __init__(
        self,
        policy,
        stats: dict,
        *,
        lr: float = 1e-3,
        steps: int = 1,
        buffer_size: int = 64,
        update_interval: int = 8,
        lambda_res: float = 1.0,
        kl_eps: float = 1e-8,
    ):
        self.policy = policy
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.basis = torch.tensor(stats["basis"], dtype=torch.float32)  # [K, D]
        self.lambdas = torch.tensor(stats["lambdas"], dtype=torch.float32)  # [K]
        self.tau = float(stats["tau"])
        self.rank = int(stats["rank"])
        self.lr = lr
        self.steps = steps
        self.buffer_size = buffer_size
        self.lambda_res = lambda_res
        self.kl_eps = kl_eps
        self.update_interval = max(1, int(update_interval))
        self.calls = 0
        probes = stats.get("probes")
        weights = stats.get("probe_weights")
        self.probes = torch.tensor(probes, dtype=torch.float32) if probes else self._probe_bank(self.rank)
        self.probe_weights = torch.tensor(weights, dtype=torch.float32) if weights else torch.ones(self.probes.shape[0])
        self.buffer: list[dict] = []
        self.parameters = self._select_parameters()
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr, weight_decay=0.0)

    def _select_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for module in self.policy.representation_extractor.modules():
            if isinstance(module, nn.LayerNorm) and module.weight is not None:
                params.append(module.weight)
                if module.bias is not None:
                    params.append(module.bias)
        if not params:
            raise RuntimeError("no LayerNorm affine parameters found")
        return params

    def _probe_bank(self, rank: int) -> torch.Tensor:
        probes = [torch.eye(rank)[i] for i in range(rank)]
        for i in range(rank):
            for j in range(i + 1, rank):
                probes.append((torch.eye(rank)[i] + torch.eye(rank)[j]) / math.sqrt(2))
                probes.append((torch.eye(rank)[i] - torch.eye(rank)[j]) / math.sqrt(2))
        return torch.stack(probes)  # [K^2, K]

    def _feature(self, batch: dict) -> torch.Tensor:
        representation = self.policy.representation_extractor(
            batch["images"], batch["state"], batch["instruction"]
        )
        tokens = representation.tokens  # [B, T, D]
        return tokens.mean(dim=1)  # per-sample feature, batch statistics stay meaningful

    def _gaussian_kl(self, m1, v1, m2, v2):
        """Exact official SSA loss (utils/loss.py) with dim_reduction='none'."""
        return (v2.log() - v1.log() + (v1 + (m2 - m1).square()) / (v2 + self.kl_eps) - 1) / 2

    def _loss(self, features: torch.Tensor) -> torch.Tensor:
        device = features.device
        mean = self.mean.to(device)
        basis = self.basis.to(device)
        lambdas = self.lambdas.to(device)
        probes = self.probes.to(device)
        weights = self.probe_weights.to(device)
        centred = features - mean
        u = centred @ basis.T
        residual = centred - u @ basis
        p = u @ probes.T
        mu_q = p.mean(dim=0)
        var_q = p.var(dim=0)  # unbiased, as in the official implementation
        var_source = (probes ** 2 * lambdas.unsqueeze(0)).sum(dim=1)
        zeros = torch.zeros_like(mu_q)
        kl = self._gaussian_kl(mu_q, var_q, zeros, var_source) + self._gaussian_kl(
            zeros, var_source, mu_q, var_q
        )
        loss_sup = (kl * weights).mean()
        dim_res = residual.shape[1]
        mu_perp = residual.mean(dim=0)
        nu_perp = residual.var(dim=0).mean()
        tau = torch.full_like(nu_perp, max(self.tau, self.kl_eps))
        zeros_res = torch.zeros_like(mu_perp)
        kl_res = self._gaussian_kl(mu_perp, nu_perp, zeros_res, tau) + self._gaussian_kl(
            zeros_res, tau, mu_perp, nu_perp
        )
        loss_res = kl_res.sum() / max(dim_res, 1)
        return loss_sup + self.lambda_res * loss_res

    def update(self, batch: dict) -> float | None:
        with torch.enable_grad():
            features = self._feature(batch)
            loss = self._loss(features)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
            self.optimizer.step()
        return float(loss.detach())
