"""Token-level style-statistics modulation for the learned-query resampler.

Motivation
----------
The channel-level hard gate was shown to select domain-sensitive dimensions that
are not action-causal (notes/43, notes/47).  This module instead leaves the
token content intact and replaces its first/second-order feature statistics with
those of a donor sample:

    z_norm = (z - mu) / sigma
    z_mod  = z_norm * sigma_donor + mu_donor

`random_stats` uses an arbitrary donor.  `cross_style_stats` prefers a donor
from a different style/domain.  The optional action-aware variant reduces the
mixing strength on resampler queries that are action-important.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class TokenImportanceEMA:
    """Per-resampler-query action importance from a clean forward/backward.

    v_k = EMA( mean_{batch, views, dim} |token_k * dL/dtoken_k| )
    """

    def __init__(self, num_tokens: int, momentum: float = 0.9):
        self.num_tokens = int(num_tokens)
        self.momentum = float(momentum)
        self.value = torch.zeros(self.num_tokens)
        self._seen = False

    @torch.no_grad()
    def update_from_tensor(self, tokens: torch.Tensor) -> None:
        """tokens: (N, K, D) with ``.grad`` populated after backward."""
        if tokens.grad is None:
            return
        g = tokens.grad.detach().float()
        f = tokens.detach().float()
        if g.shape != f.shape or g.numel() == 0:
            return
        score = (f * g).abs().mean(dim=(0, 2))
        if not torch.isfinite(score).all() or float(score.max()) <= 0:
            return
        score = score.cpu()
        if self._seen:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * score
        else:
            self.value = score
            self._seen = True

    def normalized(self) -> torch.Tensor:
        """Min-max normalize to [0, 1]; zero before the first update."""
        if not self._seen or self.value.numel() == 0:
            return torch.zeros(self.num_tokens)
        lo = self.value.min()
        hi = self.value.max()
        return ((self.value - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)

    def ready(self) -> bool:
        return self._seen

    def state_dict(self) -> Dict[str, torch.Tensor | bool | float]:
        return {"value": self.value.detach().cpu(),
                "seen": bool(self._seen),
                "momentum": float(self.momentum)}

    def load_state_dict(self, sd: Dict) -> None:
        if "value" in sd:
            self.value = sd["value"].detach().cpu()
        if "seen" in sd:
            self._seen = bool(sd["seen"])
        if "momentum" in sd:
            self.momentum = float(sd["momentum"])

    def summary(self) -> Dict[str, float | bool]:
        if not self._seen or self.value.numel() == 0:
            return {"ready": False, "mean": 0.0, "std": 0.0, "max": 0.0}
        return {"ready": True,
                "mean": float(self.value.mean()),
                "std": float(self.value.std()) if self.value.numel() > 1 else 0.0,
                "max": float(self.value.max())}


class TokenStyleModulator(nn.Module):
    """AdaIN-style token statistics modulation with optional action protection."""

    MODES = ("none", "random_stats", "cross_style_stats",
             "cross_style_stats_action_aware")

    def __init__(self, mode: str = "none", alpha: float = 0.5,
                 action_weight: float = 1.0, protect_ratio: float = 0.0,
                 eps: float = 1e-5):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown token_style_mode={mode}")
        self.mode = str(mode)
        self.alpha = float(alpha)
        self.action_weight = float(action_weight)
        self.protect_ratio = float(protect_ratio)
        self.eps = float(eps)
        # K-length buffer; empty until the first action-importance update.
        self.register_buffer("action_importance", torch.zeros(0), persistent=False)
        self.last_stats_t: Dict[str, torch.Tensor] = {}
        self.last_donor: torch.Tensor | None = None

    def set_action_importance(self, value: torch.Tensor | None) -> None:
        if value is None:
            self.action_importance = torch.zeros(0, device=self.action_importance.device)
        else:
            self.action_importance = value.detach().float()

    def summary(self) -> Dict[str, float | str]:
        out: Dict[str, float | str] = {"mode": self.mode, "alpha": self.alpha,
                                       "protect_ratio": self.protect_ratio}
        for k, v in self.last_stats_t.items():
            out[k] = float(v)
        return out

    def forward(self, tokens: torch.Tensor, domain_ids: torch.Tensor | None = None,
                training: bool = True) -> torch.Tensor:
        """tokens: (B, V, K, D) -> (B, V, K, D)."""
        if not training or self.mode == "none" or self.alpha <= 0:
            return tokens
        B, V, K, D = tokens.shape
        z = tokens.reshape(B * V, K, D)
        n = z.shape[0]
        mu = z.mean(dim=1)                                      # (N, D)
        sigma = z.std(dim=1, unbiased=False).clamp_min(self.eps)

        d = None
        if domain_ids is not None:
            d = domain_ids.detach().view(-1).to(device=z.device, dtype=torch.long)
            if d.numel() != B:
                if d.numel() == 0:
                    d = None
                else:
                    d = d.repeat((B + d.numel() - 1) // d.numel())[:B]
        row_dom = (d[:, None].repeat(1, V).reshape(-1)
                   if d is not None else None)

        donor = torch.randint(0, n, (n,), device=z.device)
        if (self.mode in ("cross_style_stats", "cross_style_stats_action_aware")
                and row_dom is not None and n > 1):
            same = row_dom[donor] == row_dom
            # Prefer a different-domain donor; a few resamples are enough for
            # the 10-style × 3-sample batches used in the mechanism probes.
            for _ in range(8):
                if not same.any():
                    break
                idx = torch.nonzero(same, as_tuple=False).view(-1)
                donor[idx] = torch.randint(0, n, (idx.numel(),), device=z.device)
                same = row_dom[donor] == row_dom
        if n > 1:
            self_idx = torch.arange(n, device=z.device)
            donor = torch.where(donor == self_idx, (donor + 1) % n, donor)

        self.last_donor = donor.detach()
        mu_d = mu[donor]
        sigma_d = sigma[donor]
        z_mod = (z - mu[:, None, :]) / sigma[:, None, :] \
            * sigma_d[:, None, :] + mu_d[:, None, :]

        alpha = torch.full((n, K, 1), self.alpha, device=z.device, dtype=z.dtype)
        if self.mode == "cross_style_stats_action_aware":
            imp = self.action_importance.to(device=z.device, dtype=z.dtype)
            if imp.numel() == K:
                alpha = alpha * (1.0 - self.action_weight
                                 * imp.clamp(0.0, 1.0)[None, :, None])
                if self.protect_ratio > 0:
                    k = max(1, min(K, int(round(self.protect_ratio * K))))
                    top = torch.topk(imp, k, largest=True).indices
                    alpha[:, top, :] = 0.0
        out = (1.0 - alpha) * z + alpha * z_mod

        # Keep diagnostics as device scalars; the policy reads them only on
        # logging steps, avoiding a per-step GPU→CPU sync.
        with torch.no_grad():
            diff_dom = (row_dom[donor] != row_dom).float().mean() \
                if row_dom is not None else z.new_zeros(())
            self.last_stats_t = {
                "mod_abs_mean": (out - z).abs().mean(),
                "stat_delta": (mu_d - mu).abs().mean(),
                "diff_domain_frac": diff_dom,
                "action_weight_mean": alpha.mean(),
            }
        return out.reshape(B, V, K, D)
