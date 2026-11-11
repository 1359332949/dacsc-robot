"""SALT adaptation port to CoRP (Phase 1, no SG-SALT innovation).

Architecture mapping (CoRP differs from the SALT paper backbones):
  * CoRP fuses Q/K/V into one Linear(768, 2304). `SelfAttention.project_qkv`
    (repos/CoRP/corp/action_generator.py:46-51) reshapes the output to
    (B, T, 3, heads, head_dim) -> index 0=Q, 1=K, 2=V, so the K slice of
    `qkv.weight` is rows [hidden_dim : 2*hidden_dim].
  * SALT updates the attention KEY projection only. Here we adapt ONLY the K
    rows of the fused qkv weight; Q, V, `projection`, and all other CoRP
    parameters stay frozen.
  * Parameterisation (SALT Appendix B.2, singular-value fine-tuning):
        W0 = U diag(sigma) V^T
        W(beta) = U diag(sigma * exp(beta)) V^T,  beta0 = 0, |beta|_inf <= eta
    with eta = 0.15, Adam, K = 5 steps, lr = 2*eta/K, 8 noise-time draws/step.
  * This is a PORT, not a claim of reproducing the paper's original backbone.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SaltKeyAdapter:
    """Adapt only the K slice of a fused qkv Linear via singular-value scaling."""

    def __init__(self, qkv: nn.Linear, hidden_dim: int, eta: float = 0.15):
        self.qkv = qkv
        w = qkv.weight.detach().float()
        if w.shape[0] != 3 * hidden_dim:
            raise ValueError(f"unexpected qkv weight shape {tuple(w.shape)}")
        self.hidden_dim = int(hidden_dim)
        self.eta = float(eta)
        self.w_qkv0 = qkv.weight.detach().clone()
        wk = w[self.hidden_dim : 2 * self.hidden_dim]              # [D, D] K slice
        u, s, vh = torch.linalg.svd(wk, full_matrices=False)
        # float64 SVD keeps the beta=0 reconstruction exact after the bf16 cast
        self.u = u.contiguous().double()
        self.s = s.contiguous().double()
        self.v = vh.contiguous().double()
        self.wk0 = wk.detach().double()          # exact beta=0 reconstruction
        self.beta = nn.Parameter(torch.zeros(self.s.numel(), device=w.device))
        self._orig_forward = qkv.forward
        qkv.forward = self._forward                              # type: ignore[assignment]

    # ---- weight construction -------------------------------------------------
    def effective_key_weight(self) -> torch.Tensor:
        # W(beta) = W0 + U diag(sigma*(exp(beta)-1)) V^T  ==  U diag(sigma*exp(beta)) V^T
        # The exp(beta)-1 form is exactly zero at beta=0 (bit-exact Frozen weight)
        # while keeping a non-zero gradient d/dbeta at beta=0.
        beta = self.beta.double().clamp(-self.eta, self.eta)
        delta = (self.u * (self.s * (torch.exp(beta) - 1.0))) @ self.v
        return (self.wk0 + delta).to(self.qkv.weight.dtype)

    def effective_qkv_weight(self) -> torch.Tensor:
        bit = self.hidden_dim
        return torch.cat(
            [self.w_qkv0[:bit], self.effective_key_weight(), self.w_qkv0[2 * bit :]], dim=0
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.effective_qkv_weight(), self.qkv.bias)

    def singular_value_ratio(self) -> tuple[float, float]:
        wk = self.effective_key_weight().detach().double()
        s0 = self.s
        s1 = torch.linalg.svdvals(wk)
        ratio = (s1 / s0).clamp(min=1e-8)
        return float(ratio.min()), float(ratio.max())

    def reset(self) -> None:
        with torch.no_grad():
            self.beta.zero_()

    def close(self) -> None:
        self.qkv.forward = self._orig_forward                    # type: ignore[assignment]


class SaltTTA:
    """Owner of the K adapter, optimizer and per-episode lifecycle."""

    def __init__(self, policy, hidden_dim: int | None = None, eta: float = 0.15,
                 adapt_steps: int = 5, draws: int = 8, lr: float | None = None):
        hidden_dim = int(hidden_dim or policy.config["hidden_dim"])
        for p in policy.parameters():
            p.requires_grad_(False)
        self.policy = policy
        self.adapter = SaltKeyAdapter(policy.action_generator.blocks[0].attention.qkv,
                                      hidden_dim, eta=eta)
        self.eta = float(eta)
        self.adapt_steps = int(adapt_steps)
        self.draws = int(draws)
        self.lr = float(lr if lr is not None else 2.0 * eta / max(adapt_steps, 1))
        self.optimizer = torch.optim.Adam([self.adapter.beta], lr=self.lr)

    def reset(self) -> None:
        self.adapter.reset()
        self.optimizer = torch.optim.Adam([self.adapter.beta], lr=self.lr)

    def close(self) -> None:
        self.adapter.close()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.adapter.beta]


def build_leftover_target(target_norm: torch.Tensor, exec_len: int, horizon: int,
                          action_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the previous normalised chunk tail to the full horizon + loss mask."""
    device, dtype = target_norm.device, target_norm.dtype
    y = torch.zeros((horizon, action_dim), device=device, dtype=dtype)
    mask = torch.zeros((horizon, 1), device=device, dtype=dtype)
    overlap = horizon - exec_len
    if overlap < 0:
        raise ValueError("exec_len larger than horizon")
    y[:overlap] = target_norm[exec_len : exec_len + overlap]
    mask[:overlap] = 1.0
    return y, mask


def flow_interpolate(y: torch.Tensor, eps: torch.Tensor, tau: torch.Tensor):
    """SALT Eq. 6: x_tau = (1-tau) y + tau eps;  v_target = eps - y."""
    x = (1.0 - tau) * y + tau * eps
    return x, eps - y


def flow_matching_leftover_loss(
    generator,
    representation,
    target_norm: torch.Tensor,
    exec_len: int,
    *,
    draws: int = 8,
    mode: str = "leftover",
) -> torch.Tensor:
    """SALT native flow-matching objective (paper Eq. 6) in normalised action space.

    target_norm: [H, A] normalised actions of the previous chunk (leftover mode
    uses target_norm[exec_len:]) or the current detached chunk (self mode).
    Only the overlap window [0, H-exec_len) is supervised for leftover mode.
    """
    h = int(generator.action_horizon)
    a = int(generator.action_dim)
    device = representation.tokens.device
    dtype = representation.action_dtype
    target = target_norm.to(device=device, dtype=dtype)
    if mode == "leftover":
        y, mask = build_leftover_target(target, exec_len, h, a)
    elif mode == "self":
        y = target[:h].clone()
        mask = torch.ones((h, 1), device=device, dtype=dtype)
    else:
        raise ValueError(f"unknown mode {mode}")
    prefix = generator.prepare_prefix(representation)
    total = torch.zeros((), device=device, dtype=torch.float32)
    for _ in range(max(int(draws), 1)):
        eps = torch.randn_like(y)
        tau = torch.rand((), device=device, dtype=dtype)
        x, v_target = flow_interpolate(y, eps, tau)
        v = generator(x.unsqueeze(0), tau.expand(1), prefix)[0]
        diff = (v - v_target) ** 2
        total = total + (diff * mask).mean()
    return total / max(int(draws), 1)


class FeatureChangeGate:
    """SALT-style gate on per-token visual change (agentview tokens).

    Calibration: collect per-token cosine changes on nominal (clean) replans,
    keep token-wise median and MAD, threshold = median + z * MAD with a floor on
    the MAD scale and clipping of the score. The exact paper constants for the
    MAD floor / clipping / z must be verified against SALT Appendix B.4; they are
    exposed as constructor arguments and recorded in the run config.
    Once triggered, the gate latches for the rest of the episode.
    """

    def __init__(self, z: float = 4.0, mad_floor: float = 1e-3, clip: float = 5.0,
                 num_tokens: int = 48):
        self.z = float(z)
        self.mad_floor = float(mad_floor)
        self.clip = float(clip)
        self.num_tokens = int(num_tokens)
        self.median: torch.Tensor | None = None
        self.mad: torch.Tensor | None = None
        self.prev: torch.Tensor | None = None
        self.active = False

    def token_change(self, tokens: torch.Tensor) -> torch.Tensor:
        view = tokens[0, : self.num_tokens].float()
        if self.prev is None:
            self.prev = view.detach()
            return torch.zeros(self.num_tokens, device=tokens.device)
        cos = torch.nn.functional.cosine_similarity(view, self.prev, dim=-1)
        self.prev = view.detach()
        return (1.0 - cos).clamp(0.0, self.clip)

    def calibrate(self, changes: list[torch.Tensor]) -> None:
        mat = torch.stack([c.detach().cpu() for c in changes])
        self.median = mat.median(dim=0).values
        self.mad = (mat - self.median).abs().median(dim=0).values.clamp(min=self.mad_floor)

    def score(self, change: torch.Tensor) -> float:
        c = change.detach().cpu()
        if self.median is None or self.mad is None:
            return float(c.median())
        z = ((c - self.median) / self.mad).median()
        return float(z)

    def update(self, tokens: torch.Tensor) -> bool:
        change = self.token_change(tokens)
        if self.active:
            return True
        if self.median is not None and self.score(change) > self.z:
            self.active = True
        return self.active

    def reset(self) -> None:
        self.prev = None
        self.active = False
