"""Full-write mean alignment at the action interface.

Audit result: the domain shift is clearly identifiable at the write level
(between-domain mean distance 3.22 vs within-domain std 0.23, SNR ~14), but our
previous modules corrected only the residual or only the action-null component.
This module tests the simplest correction: align the FULL write mean to the
source mean (bounded, last Euler step only):

    m_t = EMA_t mean_tokens(w)
    w'  = w - clip(m_t - mu_s, c_max)

No labels; source mu_s comes from trajectory-matched clean statistics.
"""

from __future__ import annotations

import torch


class FullMeanAlignment:
    def __init__(
        self,
        attention,
        stats: dict,
        *,
        num_steps: int = 10,
        action_horizon: int = 12,
        ema: float = 0.9,
        warmup: int = 2,
        c_max: float = 3.0,
        shift_min: float = 0.5,
    ):
        self.attention = attention
        self.original = attention.attend
        self.mean_s = torch.tensor(stats["mean"], dtype=torch.float32)
        self.num_steps = int(num_steps)
        self.action_horizon = int(action_horizon)
        self.ema = float(ema)
        self.warmup = int(warmup)
        self.c_max = float(c_max)
        self.shift_min = float(shift_min)
        self.step = 0
        self.steps = 0
        self.mean_t: torch.Tensor | None = None
        self.correction_norm = 0.0
        self.shift_norm = 0.0
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
        batch_mean = output.reshape(-1, output.shape[-1]).mean(dim=0).detach()
        self.mean_t = batch_mean if self.mean_t is None else self.ema * self.mean_t + (1 - self.ema) * batch_mean
        self.steps += 1
        if self.steps <= self.warmup:
            self.mode_counts["warmup"] += 1
            return output
        delta = self.mean_t - mean_s
        self.shift_norm = float(delta.norm())
        if self.shift_norm < self.shift_min:
            self.mode_counts["skip"] += 1
            return output
        correction = -delta
        if self.shift_norm > self.c_max:
            correction = correction * (self.c_max / self.shift_norm)
        self.correction_norm = float(correction.norm())
        self.mode_counts["align"] += 1
        return output + correction

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
