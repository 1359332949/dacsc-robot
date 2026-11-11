"""State-conditioned source alignment at the action interface.

Problem found by auditing the unconditional versions: the per-frame write mean
fluctuates with the robot state (std ~5-7) far more than the domain difference
(~1.9), so unconditional alignment mostly injects state noise.

Fix: model the write as a function of an observable conditioning signal phi
(mean of the attention query over heads/tokens = action-context summary),
estimate the SOURCE conditional mean with ridge regression on clean calibration
pairs, estimate the TARGET conditional mean online from the current episode's
own frames, and correct only the residual domain-specific difference:

    phi_n = (phi - mu_phi) / sd_phi
    g_s(phi) = [1, phi_n] B_s          (offline, clean data)
    g_t(phi) = [1, phi_n] B_t          (online ridge, target episode)
    w' = w - clip(g_t(phi) - g_s(phi), c_max)

Applied at the last Euler step only; bounded, additive, reversible; no labels.
"""

from __future__ import annotations

import numpy as np
import torch


class ConditionalAlignment:
    def __init__(
        self,
        attention,
        calibration_path: str,
        *,
        num_steps: int = 10,
        action_horizon: int = 12,
        ridge: float = 1.0,
        min_samples: int = 24,
        refit_interval: int = 8,
        c_max: float = 3.0,
        ema: float = 0.0,
    ):
        self.attention = attention
        self.original = attention.attend
        data = np.load(calibration_path)
        phi = data["phi"].astype(np.float64)
        write = data["write"].astype(np.float64)
        self.mu_phi = phi.mean(axis=0)
        self.sd_phi = phi.std(axis=0) + 1e-6
        phi_n = (phi - self.mu_phi) / self.sd_phi
        X = np.concatenate([np.ones((phi_n.shape[0], 1)), phi_n], axis=1)
        reg = ridge * np.eye(X.shape[1])
        reg[0, 0] = 0.0
        self.b_s = np.linalg.solve(X.T @ X + reg, X.T @ write)          # [d+1, D]
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
        self.correction_norm = 0.0
        self.mode_counts = {"cond": 0, "skip": 0, "warmup": 0}
        attention.attend = self._attend  # type: ignore[assignment]

    def _fit_target(self) -> None:
        phi = np.stack(self.phi_buf).astype(np.float64)
        write = np.stack(self.write_buf).astype(np.float64)
        phi_n = (phi - self.mu_phi) / self.sd_phi
        X = np.concatenate([np.ones((phi_n.shape[0], 1)), phi_n], axis=1)
        reg = np.eye(X.shape[1])
        reg[0, 0] = 0.0
        self.b_t = np.linalg.solve(X.T @ X + reg, X.T @ write)

    def _predict(self, b: np.ndarray, phi_n: np.ndarray) -> np.ndarray:
        x = np.concatenate([np.ones(1), phi_n])
        return x @ b

    def _attend(self, *args, **kwargs):
        query = args[0]
        output = self.original(*args, **kwargs)
        if output.shape[1] != self.action_horizon:
            self.step = 0
            return output
        self.step += 1
        if self.step != self.num_steps:
            return output
        phi = query.detach().mean(dim=(1, 2)).float().cpu().numpy().reshape(-1)
        write = output.detach().reshape(-1, output.shape[-1]).mean(dim=0).float().cpu().numpy()
        self.phi_buf.append(phi)
        self.write_buf.append(write)
        self.count += 1
        if self.count < self.min_samples:
            self.mode_counts["warmup"] += 1
            return output
        if self.b_t is None or (self.count % self.refit_interval == 0):
            self._fit_target()
        phi_n = (phi - self.mu_phi) / self.sd_phi
        delta = self._predict(self.b_t, phi_n) - self._predict(self.b_s, phi_n)
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
