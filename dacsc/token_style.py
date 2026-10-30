"""Token-level style-statistics modulation for a token sequence (patch or resampler).

Motivation
----------
This project adapter mixes feature statistics across samples; it is not a full
TFS-ViT implementation and does not establish that action information is preserved.
For each recipient it constructs:

    z_norm = (z - mu) / sigma
    z_mod  = z_norm * sigma_donor + mu_donor

`random_stats` uses TFS-ViT's sample permutation (self-pairs are allowed).
`cross_style_stats` samples exactly from different-style samples. For multiple
views, a sample pair shares its donor across cameras and preserves camera index.
Donors are NOT task-conditioned; this is not a same-task appearance intervention.
The optional action-aware variant reduces the
mixing strength on tokens that are action-important.

Upstream: TFS-ViT domainbed/pretrained_models/DeiT_models/models.py,
FeatureStylization.forward (MIT): detached statistics and torch.randperm(B).
Different-style eligibility and multi-view pairing are explicit adaptations.
"""

from __future__ import annotations

import warnings
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
        return {
            "value": self.value.detach().cpu(),
            "seen": bool(self._seen),
            "momentum": float(self.momentum),
        }

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
        return {
            "ready": True,
            "mean": float(self.value.mean()),
            "std": float(self.value.std()) if self.value.numel() > 1 else 0.0,
            "max": float(self.value.max()),
        }


class TokenStyleModulator(nn.Module):
    """AdaIN-style token statistics modulation with optional action protection."""

    MODES = (
        "none",
        "random_stats",
        "cross_style_stats",
        "cross_style_stats_action_aware",
        "normalize",
        "normalize_action_aware",
    )

    def __init__(
        self,
        mode: str = "none",
        alpha: float = 0.5,
        action_weight: float = 1.0,
        protect_ratio: float = 0.0,
        stats: str = "meanstd",
        dim: int | None = None,
        eps: float = 1e-5,
        donor_seed: int = 0,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown token_style_mode={mode}")
        self.mode = str(mode)
        self.alpha = float(alpha)
        self.action_weight = float(action_weight)
        self.protect_ratio = float(protect_ratio)
        if stats not in ("meanstd", "mean", "std"):
            raise ValueError(f"unknown token_style_stats={stats}")
        self.stats = str(stats)
        self.eps = float(eps)
        self.donor_seed = int(donor_seed)
        if mode in ("normalize", "normalize_action_aware"):
            if dim is None:
                raise ValueError("normalize mode requires dim")
            self.norm_gamma = nn.Parameter(torch.ones(int(dim)))
            self.norm_beta = nn.Parameter(torch.zeros(int(dim)))
        # K-length buffer; empty until the first action-importance update.
        self.register_buffer("action_importance", torch.zeros(0), persistent=False)
        self.last_stats_t: Dict[str, torch.Tensor] = {}
        self.last_donor: torch.Tensor | None = None
        # 独立 RNG：donor 采样不能污染 flow-matching 的全局 RNG（paired 协议）。
        self._donor_gen: torch.Generator | None = None

    def _generator(self, device) -> torch.Generator:
        if self._donor_gen is None or self._donor_gen.device != device:
            self._donor_gen = torch.Generator(device=device)
            self._donor_gen.manual_seed(self.donor_seed)
        return self._donor_gen

    def set_action_importance(self, value: torch.Tensor | None) -> None:
        if value is None:
            self.action_importance = torch.zeros(
                0, device=self.action_importance.device
            )
        else:
            if value.ndim != 1:
                raise ValueError(
                    "action importance must be a one-dimensional patch-position vector"
                )
            self.action_importance = value.detach().float()

    def donor_contract(self) -> dict:
        cross_style = self.mode in (
            "cross_style_stats",
            "cross_style_stats_action_aware",
        )
        return {
            "version": "sample_pair_v2",
            "task_conditioned": False,
            "pairing": (
                "different_style_with_replacement"
                if cross_style
                else "batch_permutation"
            ),
            "view_pairing": "corresponding_view",
            "statistics_gradient": "detached",
        }

    def summary(self) -> Dict[str, float | str]:
        out: Dict[str, float | str] = {
            "mode": self.mode,
            "alpha": self.alpha,
            "protect_ratio": self.protect_ratio,
            "stats": self.stats,
        }
        if self.mode not in ("none", "normalize", "normalize_action_aware"):
            out["donor_rule"] = self.donor_contract()["pairing"]
        for k, v in self.last_stats_t.items():
            out[k] = float(v)
        return out

    def forward(
        self,
        tokens: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
        training: bool = True,
    ) -> torch.Tensor:
        """tokens: (B, V, K, D) -> (B, V, K, D)."""
        if self.mode == "none" or self.alpha <= 0:
            return tokens
        if not training and self.mode not in ("normalize", "normalize_action_aware"):
            return tokens
        B, V, K, D = tokens.shape
        z = tokens.reshape(B * V, K, D)
        n = z.shape[0]
        mu = z.mean(dim=1)  # (N, D)
        sigma = z.std(dim=1, unbiased=False).clamp_min(self.eps)
        if self.mode in ("normalize", "normalize_action_aware"):
            z_norm = (z - mu[:, None, :]) / sigma[:, None, :]
            out_norm = (
                z_norm * self.norm_gamma[None, None, :] + self.norm_beta[None, None, :]
            )
            keep_w = torch.zeros((1, K, 1), device=z.device, dtype=z.dtype)
            if self.mode == "normalize_action_aware":
                imp = self.action_importance.to(device=z.device, dtype=z.dtype)
                if imp.numel() != K:
                    warnings.warn(
                        f"action_importance size {imp.numel()} != K={K}; "
                        "action-aware weighting disabled",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if imp.numel() == K:
                    keep_w = (self.action_weight * imp.clamp(0.0, 1.0))[None, :, None]
                    if self.protect_ratio > 0:
                        k = max(1, min(K, int(round(self.protect_ratio * K))))
                        top = torch.topk(imp, k, largest=True).indices
                        keep_w[:, top, :] = 1.0
            out = (1.0 - keep_w) * out_norm + keep_w * z
            with torch.no_grad():
                self.last_stats_t = {
                    "mod_abs_mean": (out - z).abs().mean(),
                    "stat_delta": mu.abs().mean(),
                    "diff_domain_frac": z.new_zeros(()),
                    "action_weight_mean": (1.0 - keep_w).mean(),
                }
            return out.reshape(B, V, K, D)

        d = None
        if domain_ids is not None:
            if (
                domain_ids.ndim != 1
                or domain_ids.numel() != B
                or domain_ids.is_floating_point()
            ):
                raise ValueError(
                    "domain_ids must contain one integer native style ID per sample"
                )
            d = domain_ids.detach().to(device=z.device, dtype=torch.long)
        gen = self._generator(z.device)
        if self.mode in ("cross_style_stats", "cross_style_stats_action_aware"):
            if d is None or d.unique().numel() < 2:
                raise ValueError(
                    "cross_style_stats requires >=2 native styles in the batch; no random fallback"
                )
            # Exact conditional sampling with PyTorch; unlike bounded rejection,
            # this stays cross-style even for highly imbalanced style counts.
            eligible = (d[:, None] != d[None, :]).float()
            sample_donor = torch.multinomial(eligible, 1, generator=gen).squeeze(1)
        else:
            sample_donor = torch.randperm(B, device=z.device, generator=gen)
        donor = (sample_donor[:, None] * V + torch.arange(V, device=z.device)).reshape(
            -1
        )
        row_dom = None if d is None else d.repeat_interleave(V)
        self.last_donor = donor.detach()
        # TFS-ViT / MixStyle detach statistics; gradients flow through normalized
        # recipient content, not across samples through donor mean/std.
        mu, sigma = mu.detach(), sigma.detach()
        mu_d = mu[donor]
        sigma_d = sigma[donor]
        z_norm = (z - mu[:, None, :]) / sigma[:, None, :]
        if self.stats == "meanstd":
            z_mod = z_norm * sigma_d[:, None, :] + mu_d[:, None, :]
        elif self.stats == "mean":
            z_mod = z - mu[:, None, :] + mu_d[:, None, :]
        else:  # std only
            z_mod = z_norm * sigma_d[:, None, :] + mu[:, None, :]

        alpha = torch.full((n, K, 1), self.alpha, device=z.device, dtype=z.dtype)
        if self.mode == "cross_style_stats_action_aware":
            imp = self.action_importance.to(device=z.device, dtype=z.dtype)
            if imp.numel() not in (0, K):
                raise ValueError(
                    "action importance length differs from the current patch-token count"
                )
            if imp.numel() == K:
                alpha = alpha * (
                    1.0 - self.action_weight * imp.clamp(0.0, 1.0)[None, :, None]
                )
                if self.protect_ratio > 0:
                    k = max(1, min(K, int(round(self.protect_ratio * K))))
                    top = torch.topk(imp, k, largest=True).indices
                    alpha[:, top, :] = 0.0
        out = (1.0 - alpha) * z + alpha * z_mod

        # Keep diagnostics as device scalars; the policy reads them only on
        # logging steps, avoiding a per-step GPU→CPU sync.
        with torch.no_grad():
            diff_dom = (
                (row_dom[donor] != row_dom).float().mean()
                if row_dom is not None
                else z.new_zeros(())
            )
            self.last_stats_t = {
                "mod_abs_mean": (out - z).abs().mean(),
                "stat_delta": (mu_d - mu).abs().mean(),
                "diff_domain_frac": diff_dom,
                "action_weight_mean": alpha.mean(),
                "self_donor_frac": (sample_donor == torch.arange(B, device=z.device))
                .float()
                .mean(),
            }
        return out.reshape(B, V, K, D)
