"""Source-calibrated, target-only visual bias estimator.

Source/development data fits a conditional visual baseline g_src(p_t, l).
At deployment only target observations and history are used:
    e_t = V_t - g_src(p_t, l)
    delta_t = EMA(e_0:t), clipped to a bounded norm
    V'_t = V_t - alpha * delta_t
No target Clean feature, target success label, or MuJoCo object truth is read.
"""

from __future__ import annotations
import numpy as np
import torch


class ConditionalSourceBaseline:
    def __init__(self, l2: float = 1.0, standardize: bool = True):
        self.l2 = float(l2)
        self.standardize = bool(standardize)
        self.mean = None
        self.scale = None
        self.coef = None
        self.intercept = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("x/y must be [N,D] with equal N")
        if self.standardize:
            self.mean = x.mean(0)
            self.scale = x.std(0); self.scale[self.scale < 1e-8] = 1.0
            z = (x - self.mean) / self.scale
        else:
            self.mean = np.zeros(x.shape[1]); self.scale = np.ones(x.shape[1])
            z = x
        z = np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)
        # Ridge with an unpenalized intercept.
        gram = z.T @ z
        reg = self.l2 * np.eye(gram.shape[0]); reg[-1, -1] = 0.0
        self.coef = np.linalg.solve(gram + reg, z.T @ y)
        self.intercept = self.coef[-1]
        self.coef = self.coef[:-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("fit must be called first")
        x = np.asarray(x, dtype=np.float64)
        z = (x - self.mean) / self.scale if self.standardize else x
        return z @ self.coef + self.intercept


class TargetOnlyMeanEstimator:
    def __init__(self, source_baseline: ConditionalSourceBaseline, momentum: float = 0.5,
                 alpha: float = 1.0, clip_norm: float | None = None):
        self.source = source_baseline
        self.momentum = float(momentum)
        self.alpha = float(alpha)
        self.clip_norm = clip_norm
        self.ema = None

    def reset(self) -> None:
        self.ema = None

    @torch.no_grad()
    def residual(self, visual: torch.Tensor, context: np.ndarray) -> torch.Tensor:
        """visual: [B,N,D] or [B,D]; context: [B,C]."""
        if self.source.coef is None:
            raise RuntimeError("source baseline is not fitted")
        if visual.ndim == 3:
            pooled = visual.mean(dim=1)
        elif visual.ndim == 2:
            pooled = visual
        else:
            raise ValueError("visual must be [B,N,D] or [B,D]")
        ctx = context.detach().cpu().numpy() if torch.is_tensor(context) else np.asarray(context)
        prediction = torch.as_tensor(self.source.predict(ctx),
                                     device=pooled.device, dtype=pooled.dtype)
        return pooled - prediction

    @torch.no_grad()
    def update(self, visual: torch.Tensor, context: np.ndarray) -> torch.Tensor:
        e = self.residual(visual, context)
        if self.ema is None:
            self.ema = e.clone()
        else:
            self.ema = self.momentum * self.ema + (1.0 - self.momentum) * e
        return self.ema

    @torch.no_grad()
    def correction(self, visual: torch.Tensor, context: np.ndarray) -> torch.Tensor:
        delta = self.update(visual, context).clone()
        if self.clip_norm is not None:
            norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            delta = delta * (self.clip_norm / norm).clamp(max=1.0)
        return self.alpha * delta
