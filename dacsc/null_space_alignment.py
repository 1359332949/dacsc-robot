"""Action-Null-Space test-time adaptation (ANST).

Empirical basis (this project):
  * paired clean/corrupted writes differ by ||dw||~7, of which ~39% lies in the
    action-sensitive subspace A (the policy's own compensation; success episodes
    have significantly larger A-shift than failures);
  * the residual-mean shift our earlier modules corrected lies ~91% in the action
    NULL space and has no measurable behavioural effect;
  * aligning A directly is harmful (-15.9pp, p=0.0043) because it removes the
    policy's compensation.

Theory: GPM (gradient projection memory, arXiv:2103.09762) shows that updates
projected into the null space of important gradients do not interfere with the
protected function; CORAL/SSA-style moment matching reduces the distribution
divergence term (Ben-David risk bound). ANST therefore aligns only the action-null
component of the write-mean shift:

    P_N = I - Q_A Q_A^T                (Q_A: orthonormal action-sensitive basis)
    c   = -alpha * P_N (m_t - mu_s)    (bounded by c_max)
    w'  = w + c                        (only at the last Euler step)

No labels are used; source statistics and A come from clean source data.
"""

from __future__ import annotations

import torch


class NullSpaceAlignment:
    def __init__(
        self,
        attention,
        stats: dict,
        *,
        num_steps: int = 10,
        action_horizon: int = 12,
        alpha: float = 1.0,
        ema: float = 0.9,
        warmup: int = 2,
        c_max: float = 3.0,
        shift_min: float = 0.5,
        rank_ratio: float = 1e-8,
    ):
        self.attention = attention
        self.original = attention.attend
        self.mean_s = torch.tensor(stats["mean"], dtype=torch.float32)
        action = torch.tensor(stats["action_subspace"], dtype=torch.float32)   # [D, r]
        u, s, _ = torch.linalg.svd(action, full_matrices=False)
        r = int(torch.sum(s > s[0] * rank_ratio)) if s.numel() else 0
        self.q_a = u[:, :r].contiguous()                                       # [D, r]
        self.num_steps = int(num_steps)
        self.action_horizon = int(action_horizon)
        self.alpha = float(alpha)
        self.ema = float(ema)
        self.warmup = int(warmup)
        self.c_max = float(c_max)
        self.shift_min = float(shift_min)
        self.step = 0
        self.steps = 0
        self.mean_t: torch.Tensor | None = None
        self.correction_norm = 0.0
        self.null_frac = 0.0
        self.mode_counts = {"align": 0, "skip": 0, "warmup": 0}
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
        mean_s = self.mean_s.to(device)
        q_a = self.q_a.to(device)
        batch_mean = output.reshape(-1, output.shape[-1]).mean(dim=0).detach()
        self.mean_t = batch_mean if self.mean_t is None else self.ema * self.mean_t + (1 - self.ema) * batch_mean
        self.steps += 1
        if self.steps <= self.warmup:
            self.mode_counts["warmup"] += 1
            return output
        delta = self.mean_t - mean_s
        # null-space component: remove the action-sensitive part
        delta_null = delta - q_a @ (q_a.T @ delta)
        self.null_frac = float(delta_null.norm() / (delta.norm() + 1e-9))
        if float(delta_null.norm()) < self.shift_min:
            self.mode_counts["skip"] += 1
            return output
        correction = -self.alpha * delta_null
        if float(correction.norm()) > self.c_max:
            correction = correction * (self.c_max / float(correction.norm()))
        self.correction_norm = float(correction.norm())
        self.mode_counts["align"] += 1
        return output + correction

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
