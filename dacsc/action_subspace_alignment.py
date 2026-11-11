"""Action-sensitive mean alignment at the action interface.

Motivation (paired experiment, same task/actions, clean vs corrupted):
    ||w_corrupt - w_clean|| ~ 6.5-7.5 at the last Euler step, and **~39%** of that
    difference lies in the action-sensitive subspace A.
    The residual-mean shift that our previous modules corrected lies only ~9% in A.
=> the actionable part of the shift is the *action-subspace component of the write
   mean*, so we align exactly that component to the source write mean:

    m_t = EMA of mean_tokens(w)                  (full write, not residual)
    c   = -alpha * P_A (m_t - mu_s)              (P_A: projector on action subspace)
    w'  = w + c                                  (bounded by |c| <= c_max)

Everything else (the action-null part) is left untouched; the transform is an
additive, reversible, bounded correction applied only at the last Euler step.
Source mu_s and A come from clean source statistics; no labels are used.
"""

from __future__ import annotations

import torch


class ActionSubspaceAlignment:
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
        rank: int | None = None,
    ):
        self.attention = attention
        self.original = attention.attend
        self.mean_s = torch.tensor(stats["mean"], dtype=torch.float32)
        action = torch.tensor(stats["action_subspace"], dtype=torch.float32)   # [D, r]
        u, s, _ = torch.linalg.svd(action, full_matrices=False)
        r = int(rank) if rank is not None else int(torch.sum(s > s[0] * 1e-8))
        self.q = u[:, :r].contiguous()                                         # [D, r]
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
        q = self.q.to(device)
        batch_mean = output.reshape(-1, output.shape[-1]).mean(dim=0).detach()
        self.mean_t = batch_mean if self.mean_t is None else self.ema * self.mean_t + (1 - self.ema) * batch_mean
        self.steps += 1
        if self.steps <= self.warmup:
            self.mode_counts["warmup"] += 1
            return output
        delta = self.mean_t - mean_s
        projected = q @ (q.T @ delta)
        if float(projected.norm()) < self.shift_min:
            self.mode_counts["skip"] += 1
            return output
        correction = -self.alpha * projected
        if float(correction.norm()) > self.c_max:
            correction = correction * (self.c_max / float(correction.norm()))
        self.correction_norm = float(correction.norm())
        self.mode_counts["align"] += 1
        return output + correction

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
