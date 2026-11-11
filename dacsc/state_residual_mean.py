"""State-conditional residual-mean alignment.

Two-stage estimator that avoids fitting a full online regression (which was too
noisy with ~20 frames):
  1. offline: ridge g_s(phi) on clean pairs, phi = [proprioception, step_norm];
  2. online : residual e_i = w_i - g_s(phi_i); running mean of e is the estimated
     state-conditional domain shift; correction = -clip(mean_e, c_max).

On clean data E[w|phi] = g_s(phi) up to calibration error, so mean_e ~ 0 and the
correction vanishes (safety); under shift it estimates the conditional shift with
estimator variance sigma/sqrt(n) instead of parameter noise.
"""

from __future__ import annotations

import numpy as np
import torch


class StateResidualMeanAlignment:
    def __init__(
        self,
        attention,
        calibration_path: str,
        *,
        num_steps: int = 10,
        action_horizon: int = 12,
        ridge: float = 1.0,
        warmup: int = 4,
        c_max: float = 2.0,
        shift_min: float = 0.3,
        ema: float = 0.9,
    ):
        self.attention = attention
        self.original = attention.attend
        data = np.load(calibration_path)
        phi = data["phi"].astype(np.float64)
        write = data["write"].astype(np.float64)
        self.mu_phi = phi.mean(axis=0)
        self.sd_phi = phi.std(axis=0) + 1e-6
        x = (phi - self.mu_phi) / self.sd_phi
        X = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        reg = ridge * np.eye(X.shape[1]); reg[0, 0] = 0.0
        self.b_s = np.linalg.solve(X.T @ X + reg, X.T @ write)
        self.num_steps = int(num_steps)
        self.action_horizon = int(action_horizon)
        self.warmup = int(warmup)
        self.c_max = float(c_max)
        self.shift_min = float(shift_min)
        self.ema = float(ema)
        self.step = 0
        self.steps = 0
        self.mean_e: np.ndarray | None = None
        self.conditioning: np.ndarray | None = None
        self.correction_norm = 0.0
        self.shift_norm = 0.0
        self.mode_counts = {"align": 0, "skip": 0, "warmup": 0}
        attention.attend = self._attend  # type: ignore[assignment]

    def _g_s(self, phi: np.ndarray) -> np.ndarray:
        x = (phi - self.mu_phi) / self.sd_phi
        return np.concatenate([np.ones(1), x]) @ self.b_s

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        if output.shape[1] != self.action_horizon:
            self.step = 0
            return output
        self.step += 1
        if self.step != self.num_steps:
            return output
        if self.conditioning is None:
            return output
        w = output.detach().reshape(-1, output.shape[-1]).mean(dim=0).float().cpu().numpy()
        e = w - self._g_s(np.asarray(self.conditioning, dtype=np.float64))
        self.mean_e = e if self.mean_e is None else self.ema * self.mean_e + (1 - self.ema) * e
        self.steps += 1
        if self.steps <= self.warmup:
            self.mode_counts["warmup"] += 1
            return output
        delta = self.mean_e
        self.shift_norm = float(np.linalg.norm(delta))
        if self.shift_norm < self.shift_min:
            self.mode_counts["skip"] += 1
            return output
        correction = -delta
        if self.shift_norm > self.c_max:
            correction = correction * (self.c_max / self.shift_norm)
        self.correction_norm = float(np.linalg.norm(correction))
        self.mode_counts["align"] += 1
        return output + torch.tensor(correction, dtype=output.dtype, device=output.device)

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
