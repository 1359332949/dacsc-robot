"""Source-calibrated target-only visual residual estimation.

The source baseline and residual-content subspace are fitted from source-domain
Clean observations only. At deployment, the estimator consumes target visual
features and observable context only; target Clean features are never required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


class ConditionalSourceBaseline:
    """Ridge prediction of source-domain pooled visual features from context."""

    def __init__(self, l2: float = 1.0, standardize: bool = True):
        if not np.isfinite(l2) or l2 < 0:
            raise ValueError("l2 must be finite and non-negative")
        self.l2 = float(l2)
        self.standardize = bool(standardize)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.intercept: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("x/y must be [N,D] with equal N")
        if x.shape[0] < 2 or x.shape[1] < 1 or y.shape[1] < 1:
            raise ValueError("source fit requires at least two rows and non-empty features")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("source fit data must be finite")

        if self.standardize:
            self.mean = x.mean(axis=0)
            self.scale = x.std(axis=0)
            self.scale[self.scale < 1e-8] = 1.0
            z = (x - self.mean) / self.scale
        else:
            self.mean = np.zeros(x.shape[1], dtype=np.float64)
            self.scale = np.ones(x.shape[1], dtype=np.float64)
            z = x

        z = np.concatenate([z, np.ones((z.shape[0], 1), dtype=np.float64)], axis=1)
        gram = z.T @ z
        reg = self.l2 * np.eye(gram.shape[0], dtype=np.float64)
        reg[-1, -1] = 0.0
        try:
            solution = np.linalg.solve(gram + reg, z.T @ y)
        except np.linalg.LinAlgError as exc:
            raise ValueError("source ridge system is singular; increase l2") from exc
        self.coef = solution[:-1]
        self.intercept = solution[-1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None or self.mean is None or self.scale is None:
            raise RuntimeError("fit must be called first")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.mean.shape[0]:
            raise ValueError(f"context must be [B,{self.mean.shape[0]}]")
        if not np.isfinite(x).all():
            raise ValueError("context must be finite")
        z = (x - self.mean) / self.scale if self.standardize else x
        return z @ self.coef + self.intercept


@dataclass
class SourceResidualProjector:
    """Remove source baseline error and dominant source-content residual directions.

    The mean and PCA basis must be fitted from out-of-fold source-Clean residuals.
    This is an experimental filter: a genuine style shift may overlap the
    source-content directions and be removed as well.
    """

    mean: np.ndarray
    basis: np.ndarray

    @classmethod
    def fit(cls, source_oof_residuals: np.ndarray, rank: int) -> "SourceResidualProjector":
        residuals = np.asarray(source_oof_residuals, dtype=np.float64)
        if residuals.ndim != 2 or residuals.shape[0] < 2 or residuals.shape[1] < 1:
            raise ValueError("source residuals must be [N,D] with N >= 2")
        if not np.isfinite(residuals).all():
            raise ValueError("source residuals must be finite")
        if not isinstance(rank, int) or rank < 0 or rank >= min(residuals.shape):
            raise ValueError("rank must be in [0, min(N,D))")

        mean = residuals.mean(axis=0)
        centered = residuals - mean
        if rank == 0:
            basis = np.zeros((residuals.shape[1], 0), dtype=np.float64)
        else:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            basis = vt[:rank].T.copy()
        return cls(mean=mean, basis=basis)

    def transform(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.ndim != 2 or residual.shape[1] != self.mean.shape[0]:
            raise ValueError(f"residual must be [B,{self.mean.shape[0]}]")
        mean = torch.as_tensor(self.mean, device=residual.device, dtype=residual.dtype)
        centered = residual - mean.unsqueeze(0)
        if self.basis.shape[1] == 0:
            return centered
        basis = torch.as_tensor(self.basis, device=residual.device, dtype=residual.dtype)
        return centered - (centered @ basis) @ basis.T


def fit_source_calibration(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 10.0,
    content_rank: int = 2,
    folds: int = 5,
    seed: int = 0,
    standardize: bool = True,
) -> tuple[ConditionalSourceBaseline, SourceResidualProjector, dict]:
    """Fit source baseline and content filter from cross-fitted Clean residuals.

    Cross-fitting avoids learning the content subspace from artificially small
    in-sample residuals produced by the source baseline itself.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("x/y must be [N,D] with equal N")
    n = x.shape[0]
    if not (2 <= folds <= n):
        raise ValueError("folds must be between 2 and the number of source rows")
    if not (0 <= content_rank < min(n, y.shape[1])):
        raise ValueError("content_rank must be in [0, min(N,D))")

    rng = np.random.RandomState(seed)
    fold_ids = np.array_split(rng.permutation(n), folds)
    oof = np.empty_like(y, dtype=np.float64)
    for valid_ids in fold_ids:
        train_mask = np.ones(n, dtype=bool)
        train_mask[valid_ids] = False
        fold_model = ConditionalSourceBaseline(l2=l2, standardize=standardize)
        fold_model.fit(x[train_mask], y[train_mask])
        oof[valid_ids] = y[valid_ids] - fold_model.predict(x[valid_ids])

    baseline = ConditionalSourceBaseline(l2=l2, standardize=standardize)
    baseline.fit(x, y)
    projector = SourceResidualProjector.fit(oof, content_rank)
    singular_values = np.linalg.svd(oof - oof.mean(axis=0), compute_uv=False)
    total_energy = float(np.square(singular_values).sum())
    removed_energy = float(np.square(singular_values[:content_rank]).sum())
    diagnostics = {
        "source_rows": int(n),
        "folds": int(folds),
        "content_rank": int(content_rank),
        "oof_residual_norm_mean": float(np.linalg.norm(oof, axis=1).mean()),
        "source_residual_mean_norm": float(np.linalg.norm(projector.mean)),
        "content_energy_fraction_removed": (
            removed_energy / total_energy if total_energy > 0 else 0.0
        ),
    }
    return baseline, projector, diagnostics


class TargetOnlyMeanEstimator:
    """Episode-local EMA of target residuals; optionally filters source content."""

    def __init__(
        self,
        source_baseline: ConditionalSourceBaseline,
        momentum: float = 0.5,
        alpha: float = 1.0,
        clip_norm: float | None = None,
        residual_projector: SourceResidualProjector | None = None,
    ):
        if not np.isfinite(momentum) or not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be finite and in [0,1)")
        if not np.isfinite(alpha):
            raise ValueError("alpha must be finite")
        if clip_norm is not None and (not np.isfinite(clip_norm) or clip_norm <= 0):
            raise ValueError("clip_norm must be finite and positive")
        self.source = source_baseline
        self.momentum = float(momentum)
        self.alpha = float(alpha)
        self.clip_norm = None if clip_norm is None else float(clip_norm)
        self.residual_projector = residual_projector
        self.ema: torch.Tensor | None = None

    def reset(self) -> None:
        self.ema = None

    @torch.no_grad()
    def raw_residual(self, visual: torch.Tensor, context: np.ndarray) -> torch.Tensor:
        """Return pooled V_t - g_src(context), before source-content filtering."""
        if visual.ndim == 3:
            pooled = visual.mean(dim=1)
        elif visual.ndim == 2:
            pooled = visual
        else:
            raise ValueError("visual must be [B,N,D] or [B,D]")
        if not torch.isfinite(pooled).all():
            raise ValueError("visual features must be finite")
        ctx = context.detach().cpu().numpy() if torch.is_tensor(context) else np.asarray(context)
        if ctx.ndim == 1:
            ctx = ctx[None, :]
        if ctx.shape[0] != pooled.shape[0]:
            raise ValueError("context batch size must match visual batch size")
        prediction = torch.as_tensor(
            self.source.predict(ctx), device=pooled.device, dtype=pooled.dtype
        )
        if prediction.shape != pooled.shape:
            raise ValueError(
                f"source prediction shape {tuple(prediction.shape)} != visual {tuple(pooled.shape)}"
            )
        return pooled - prediction

    @torch.no_grad()
    def residual(self, visual: torch.Tensor, context: np.ndarray) -> torch.Tensor:
        raw = self.raw_residual(visual, context)
        if self.residual_projector is None:
            return raw
        return self.residual_projector.transform(raw)

    @torch.no_grad()
    def update(self, visual: torch.Tensor, context: np.ndarray) -> torch.Tensor:
        residual = self.residual(visual, context)
        if self.ema is None:
            self.ema = residual.clone()
        else:
            if self.ema.shape != residual.shape:
                raise ValueError("target batch shape changed; reset estimator before changing batch size")
            self.ema = self.momentum * self.ema + (1.0 - self.momentum) * residual
        return self.ema

    @torch.no_grad()
    def correction(self, visual: torch.Tensor, context: np.ndarray) -> torch.Tensor:
        delta = self.update(visual, context).clone()
        if self.clip_norm is not None:
            norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            delta = delta * (self.clip_norm / norm).clamp(max=1.0)
        return self.alpha * delta
