"""Identifiability-weighted PSC test-time adaptation.

The loss/optimizer conventions are those of the official SSA implementation
(kzkadc/regression-tta) and the PSC paper (arXiv:2603.09338):

  L_sup = mean_q  w_q * [ KL(N(mu_q, var_q) || N(0, sigma_q^2))
                        + KL(N(0, sigma_q^2) || N(mu_q, var_q)) ]
  L_res = PSC residual spectral matching (Eq. 23)
  L     = L_sup + lambda * L_res

Single documented addition: the per-probe weight w_q multiplies the PSC head
weight by an identifiability factor

  w_q = beta_q * nu_q / (nu_q + mu_hat_q^2 + eps)

where nu_q is the variability of probe q under label-preserving photometric
augmentations (estimated offline on source data). A probe direction whose
observed target shift is far larger than its nuisance variability is treated as
content (down-weighted), so alignment does not remove action-relevant structure.
"""

from __future__ import annotations

import torch

from dacsc.psc_tta import PSCTTA


class WeightedPSCTTA(PSCTTA):
    def __init__(self, policy, stats: dict, *, eps_weight: float = 1e-6, **kwargs):
        super().__init__(policy, stats, **kwargs)
        if "probe_nuisance_var" not in stats:
            raise ValueError("weighted PSC needs probe_nuisance_var in the stats file")
        self.nuisance_var = torch.tensor(stats["probe_nuisance_var"], dtype=torch.float32)
        self.eps_weight = eps_weight

    def _loss(self, features: torch.Tensor) -> torch.Tensor:
        device = features.device
        mean = self.mean.to(device)
        basis = self.basis.to(device)
        lambdas = self.lambdas.to(device)
        probes = self.probes.to(device)
        base_weights = self.probe_weights.to(device)
        nuisance_var = self.nuisance_var.to(device)
        centred = features - mean
        u = centred @ basis.T
        residual = centred - u @ basis
        p = u @ probes.T
        mu_q = p.mean(dim=0)
        var_q = p.var(dim=0)
        var_source = (probes ** 2 * lambdas.unsqueeze(0)).sum(dim=1)
        zeros = torch.zeros_like(mu_q)
        kl = self._gaussian_kl(mu_q, var_q, zeros, var_source) + self._gaussian_kl(
            zeros, var_source, mu_q, var_q
        )
        # identifiability: observed shift explained by nuisance variability
        identifiability = nuisance_var / (nuisance_var + mu_q.square() + self.eps_weight)
        weights = base_weights * identifiability
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
