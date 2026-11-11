"""SALT adaptation port to CoRP (Phase 1.1 correctness closure).

Architecture mapping: CoRP fuses Q/K/V in one Linear(768, 2304); `project_qkv`
splits (B,T,3,heads,hd) in Q,K,V order, so K rows are [768:1536]. Only those rows
are adapted via singular-value scaling; everything else is frozen.

Flow convention is CoRP-native (forward Euler, tau: 0=noise -> 1=data):
    X_tau = (1-tau) eps + tau Y
    v*    = Y - eps
This is the convention used in SALT Appendix B.3 for GR00T-style backbones and is
the one consistent with CoRP's `DiTActionGenerator.sample`.
"""

from __future__ import annotations

import torch
from torch import nn

MAD_TO_SIGMA = 1.4826
SCORE_CLIP = 10.0
FLOOR_RATIO = 0.25


class SaltKeyAdapter:
    """Adapt only the K slice of a fused qkv Linear via singular-value scaling."""

    def __init__(self, qkv: nn.Linear, hidden_dim: int, eta: float = 0.15):
        self.qkv = qkv
        self.hidden_dim = int(hidden_dim)
        self.eta = float(eta)
        w = qkv.weight.detach().float()
        if w.shape[0] != 3 * self.hidden_dim:
            raise ValueError(f"unexpected qkv weight shape {tuple(w.shape)}")
        self.w_qkv0 = qkv.weight.detach().clone()
        wk = w[self.hidden_dim : 2 * self.hidden_dim]
        u, s, vh = torch.linalg.svd(wk, full_matrices=False)
        self.u = u.contiguous().double()
        self.s = s.contiguous().double()
        self.v = vh.contiguous().double()
        self.wk0 = wk.detach().double()
        self.beta = nn.Parameter(torch.zeros(self.s.numel(), device=w.device))
        self._orig_forward = qkv.forward
        qkv.forward = self._forward                            # type: ignore[assignment]

    def effective_key_weight(self) -> torch.Tensor:
        beta = self.beta.double().clamp(-self.eta, self.eta)
        delta = (self.u * (self.s * (torch.exp(beta) - 1.0))) @ self.v
        return (self.wk0 + delta).to(self.qkv.weight.dtype)

    def effective_qkv_weight(self) -> torch.Tensor:
        d = self.hidden_dim
        return torch.cat([self.w_qkv0[:d], self.effective_key_weight(), self.w_qkv0[2 * d :]], dim=0)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.effective_qkv_weight(), self.qkv.bias)

    def singular_value_ratio(self) -> tuple[float, float]:
        s0 = self.s
        s1 = torch.linalg.svdvals(self.effective_key_weight().detach().double())
        ratio = (s1 / s0).clamp(min=1e-8)
        return float(ratio.min()), float(ratio.max())

    def diagnostics(self) -> dict:
        b = self.beta.detach().double()
        return {
            "beta_l2": float(b.norm()),
            "beta_abs_mean": float(b.abs().mean()),
            "beta_abs_max": float(b.abs().max()),
            "frac_ge_095eta": float((b.abs() >= 0.95 * self.eta).float().mean()),
            "frac_ge_099eta": float((b.abs() >= 0.99 * self.eta).float().mean()),
        }

    def reset(self) -> None:
        with torch.no_grad():
            self.beta.zero_()

    def close(self) -> None:
        self.qkv.forward = self._orig_forward                  # type: ignore[assignment]


class SaltTTA:
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


# --------------------------------------------------------------------------- #
# CoRP-native flow-matching losses
# --------------------------------------------------------------------------- #
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


def flow_interpolate(y: torch.Tensor, eps: torch.Tensor, tau: torch.Tensor,
                     convention: str = "corp"):
    """Flow interpolation in the CoRP-native direction (tau: 0=noise -> 1=data).

    Corp:  x_tau = (1-tau) eps + tau y ;  v* = y - eps
    (the previous 'salted' convention x=(1-tau)y+tau*eps is intentionally removed)
    """
    if convention != "corp":
        raise ValueError("only the CoRP-native convention is supported in this port")
    x = (1.0 - tau) * eps + tau * y
    return x, y - eps


def _masked_mse(diff: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over supervised (time, action) elements only."""
    denom = mask.sum() * diff.shape[-1]
    if float(denom) == 0:
        return torch.zeros((), device=diff.device, dtype=torch.float32)
    return (diff * mask).sum() / denom


def flow_matching_leftover_loss(
    generator,
    representation,
    target_norm: torch.Tensor,
    exec_len: int,
    *,
    draws: int = 8,
    mode: str = "leftover",
    rng: torch.Generator | None = None,
) -> torch.Tensor:
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
        eps = torch.randn(y.shape, device=device, dtype=dtype, generator=rng)
        tau = torch.rand((), device=device, dtype=dtype, generator=rng)
        x, v_target = flow_interpolate(y, eps, tau)
        v = generator(x.unsqueeze(0), tau.expand(1), prefix)[0]
        total = total + _masked_mse((v - v_target) ** 2, mask)
    return total / max(int(draws), 1)


# --------------------------------------------------------------------------- #
# SALT Appendix B.4 gate
# --------------------------------------------------------------------------- #
class FeatureChangeGate:
    """Token-wise feature-change gate with SALT B.4 statistics.

    d[r,i] = 1 - cos(F[r-1,i], F[r,i])   (after the token-norm floor)
    m_i    = median_i over nominal transitions
    s_i    = max(1.4826*MAD_i, 0.25*median_i(1.4826*MAD_i))
    S_r    = mean_i clip((d[r,i]-m_i)/s_i, 0, 10)
    threshold = 0.9 * max over held-out nominal episodes of max_r S_r (cross-fit)
    """

    def __init__(self, num_tokens: int = 96, norm_floor: float = 1e-6,
                 clip_max: float = SCORE_CLIP):
        self.num_tokens = int(num_tokens)
        self.norm_floor = float(norm_floor)
        self.clip_max = float(clip_max)
        self.median: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.threshold: float | None = None
        self.prev: torch.Tensor | None = None
        self.active = False

    def _normed(self, tokens: torch.Tensor) -> torch.Tensor:
        view = tokens[0, : self.num_tokens].float()
        norm = view.norm(dim=-1, keepdim=True).clamp_min(self.norm_floor)
        return view / norm

    def token_change(self, tokens: torch.Tensor) -> torch.Tensor:
        cur = self._normed(tokens)
        if self.prev is None:
            self.prev = cur.detach()
            return torch.zeros(self.num_tokens, device=tokens.device)
        cos = (cur * self.prev).sum(dim=-1)
        self.prev = cur.detach()
        return 1.0 - cos

    @staticmethod
    def fit_stats(changes: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mat = torch.stack([c.float() for c in changes])
        median = mat.median(dim=0).values
        mad = (mat - median).abs().median(dim=0).values
        scale = MAD_TO_SIGMA * mad
        floor = FLOOR_RATIO * scale.median()
        scale = scale.clamp_min(float(floor))
        return median, scale

    def score_raw(self, change: torch.Tensor, median: torch.Tensor | None = None,
                  scale: torch.Tensor | None = None) -> torch.Tensor:
        m = self.median if median is None else median
        s = self.scale if scale is None else scale
        if m is None or s is None:
            raise RuntimeError("gate statistics are not calibrated")
        z = ((change.float().cpu() - m) / s).clamp(0.0, self.clip_max)
        return z.mean()

    def calibrate(self, per_task: dict[int, list[list[torch.Tensor]]]) -> dict:
        """per_task[task] = list of episodes; each episode = list of token-change tensors."""
        all_changes = [c for eps in per_task.values() for ep in eps for c in ep]
        if not all_changes:
            raise ValueError("empty calibration data")
        # cross-fitted threshold: leave-one-task-out
        heldout_max: list[float] = []
        for task in per_task:
            train = [c for t, eps in per_task.items() if t != task for ep in eps for c in ep]
            if not train:
                continue
            m, s = self.fit_stats(train)
            for ep in per_task[task]:
                if not ep:
                    continue
                scores = [float(self.score_raw(c, m, s)) for c in ep]
                heldout_max.append(max(scores))
        self.threshold = 0.9 * max(heldout_max) if heldout_max else 0.0
        self.median, self.scale = self.fit_stats(all_changes)
        return {"threshold": self.threshold, "n_tokens": self.num_tokens,
                "n_transitions": len(all_changes),
                "median": self.median.tolist(), "scale": self.scale.tolist()}

    def load(self, blob: dict) -> None:
        self.median = torch.tensor(blob["median"], dtype=torch.float32)
        self.scale = torch.tensor(blob["scale"], dtype=torch.float32)
        self.threshold = float(blob["threshold"])
        self.num_tokens = int(blob.get("n_tokens", self.num_tokens))

    def update(self, tokens: torch.Tensor) -> bool:
        change = self.token_change(tokens)
        if self.active:
            return True
        if self.median is None or self.threshold is None:
            raise RuntimeError("gate is not calibrated")
        if float(self.score_raw(change)) > self.threshold:
            self.active = True
        return self.active

    def reset(self) -> None:
        self.prev = None
        self.active = False
