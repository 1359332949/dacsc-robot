"""Offline (adapt-then-freeze) state-conditional alignment.

Protocol (matches the TTA literature: adapt once, then freeze):
  warm-up: the first W predicts run WITHOUT any correction; they only collect
           (phi, write) pairs;
  estimate: e = mean_i (w_i - g_s(phi_i))   (state-conditional domain shift)
  freeze  : correction = -clip(e, c_max) and apply it for the remaining predicts.

Because the estimate comes from un-intervened frames, it is not contaminated by
the correction itself (the online versions suffered from this feedback).
"""

from __future__ import annotations

import numpy as np
import torch


class OfflineStateAlignment:
    def __init__(
        self,
        attention,
        calibration_path: str,
        *,
        num_steps: int = 10,
        action_horizon: int = 12,
        ridge: float = 1.0,
        warmup_predicts: int = 6,
        c_max: float = 3.0,
        shift_min: float = 0.3,
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
        self.warmup_predicts = int(warmup_predicts)
        self.c_max = float(c_max)
        self.shift_min = float(shift_min)
        self.step = 0
        self.predicts = 0
        self.phi_buf: list[np.ndarray] = []
        self.write_buf: list[np.ndarray] = []
        self.frozen: np.ndarray | None = None
        self.conditioning: np.ndarray | None = None
        self.shift_norm = 0.0
        self.mode_counts = {"adapt": 0, "skip": 0, "warmup": 0}
        attention.attend = self._attend  # type: ignore[assignment]

    def _g_s(self, phi: np.ndarray) -> np.ndarray:
        x = (phi - self.mu_phi) / self.sd_phi
        return np.concatenate([np.ones(1), x]) @ self.b_s

    def _estimate(self) -> None:
        phi = np.stack(self.phi_buf).astype(np.float64)
        write = np.stack(self.write_buf).astype(np.float64)
        resid = write - np.stack([self._g_s(p) for p in phi])
        e = resid.mean(axis=0)
        self.shift_norm = float(np.linalg.norm(e))
        if self.shift_norm < self.shift_min:
            self.frozen = None
        else:
            corr = -e
            if self.shift_norm > self.c_max:
                corr = corr * (self.c_max / self.shift_norm)
            self.frozen = corr

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
        self.predicts += 1
        if self.predicts <= self.warmup_predicts:
            w = output.detach().reshape(-1, output.shape[-1]).mean(dim=0).float().cpu().numpy()
            self.phi_buf.append(np.asarray(self.conditioning, dtype=np.float32))
            self.write_buf.append(w)
            self.mode_counts["warmup"] += 1
            if self.predicts == self.warmup_predicts:
                self._estimate()
            return output
        if self.frozen is None:
            self.mode_counts["skip"] += 1
            return output
        self.mode_counts["adapt"] += 1
        return output + torch.tensor(self.frozen, dtype=output.dtype, device=output.device)

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
