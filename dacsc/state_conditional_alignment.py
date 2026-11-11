"""State-conditioned alignment with domain-invariant conditioning.

phi = [proprioceptive state, normalized predict index] (domain-invariant).
Source: ridge regression w ~ [1,phi] on clean calibration pairs.
Target: online ridge on the current episode's frames.
Correction: w' = w - clip(g_t(phi) - g_s(phi), c_max), last Euler step only.

The driver must set `module.conditioning = phi` before each policy.predict call.
"""

from __future__ import annotations

import numpy as np
import torch


class StateConditionalAlignment:
    def __init__(
        self,
        attention,
        calibration_path: str,
        *,
        num_steps: int = 10,
        action_horizon: int = 12,
        ridge: float = 1.0,
        min_samples: int = 16,
        refit_interval: int = 8,
        c_max: float = 3.0,
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
        self.min_samples = int(min_samples)
        self.refit_interval = int(refit_interval)
        self.c_max = float(c_max)
        self.step = 0
        self.count = 0
        self.phi_buf: list[np.ndarray] = []
        self.write_buf: list[np.ndarray] = []
        self.b_t: np.ndarray | None = None
        self.conditioning: np.ndarray | None = None
        self.correction_norm = 0.0
        self.mode_counts = {"cond": 0, "skip": 0, "warmup": 0}
        attention.attend = self._attend  # type: ignore[assignment]

    def _fit(self) -> None:
        phi = np.stack(self.phi_buf).astype(np.float64)
        write = np.stack(self.write_buf).astype(np.float64)
        x = (phi - self.mu_phi) / self.sd_phi
        X = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        reg = np.eye(X.shape[1]); reg[0, 0] = 0.0
        self.b_t = np.linalg.solve(X.T @ X + reg, X.T @ write)

    @staticmethod
    def _pred(b: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.concatenate([np.ones(1), x]) @ b

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        if output.shape[1] != self.action_horizon:
            self.step = 0
            return output
        self.step += 1
        if self.step != self.num_steps:
            return output
        phi = self.conditioning
        if phi is None:
            return output
        write = output.detach().reshape(-1, output.shape[-1]).mean(dim=0).float().cpu().numpy()
        self.phi_buf.append(np.asarray(phi, dtype=np.float32))
        self.write_buf.append(write)
        self.count += 1
        if self.count < self.min_samples:
            self.mode_counts["warmup"] += 1
            return output
        if self.b_t is None or (self.count % self.refit_interval == 0):
            self._fit()
        x = (np.asarray(phi, dtype=np.float64) - self.mu_phi) / self.sd_phi
        delta = self._pred(self.b_t, x) - self._pred(self.b_s, x)
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            self.mode_counts["skip"] += 1
            return output
        if norm > self.c_max:
            delta = delta * (self.c_max / norm)
        correction = torch.tensor(-delta, dtype=output.dtype, device=output.device)
        self.correction_norm = float(torch.linalg.norm(correction))
        self.mode_counts["cond"] += 1
        return output + correction

    def reset(self):
        self.step = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]
