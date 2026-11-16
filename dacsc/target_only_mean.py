"""Source-calibrated target-only visual residual estimation.

The source baseline and optional content filter are fitted from source-domain
Clean observations only. A source-calibrated confidence gate abstains when the
target residual looks like ordinary source residual noise or the proprioceptive
context is outside the support of source calibration data.
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
    """Remove dominant source-content residual directions.

    A genuine visual-style shift can overlap these directions, so this remains
    an ablation rather than a guaranteed content/style decomposition.
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


@dataclass
class SourceResidualConfidenceGate:
    """Source-only residual novelty × source-context support confidence.

    Calibrated from out-of-fold source-Clean residuals. It abstains on residuals
    within the source noise envelope and outside source-context support.
    This is a conservative heuristic, not a learned content/style separator.
    """

    source_contexts_z: np.ndarray
    context_mean: np.ndarray
    context_scale: np.ndarray
    support_radius: float
    residual_center: np.ndarray
    residual_threshold: float
    support_outer_multiplier: float = 2.0

    @classmethod
    def fit(
        cls,
        source_contexts: np.ndarray,
        source_oof_residuals: np.ndarray,
        *,
        residual_quantile: float = 0.95,
        support_quantile: float = 0.95,
        support_outer_multiplier: float = 2.0,
    ) -> "SourceResidualConfidenceGate":
        x = np.asarray(source_contexts, dtype=np.float64)
        r = np.asarray(source_oof_residuals, dtype=np.float64)
        if x.ndim != 2 or r.ndim != 2 or x.shape[0] != r.shape[0]:
            raise ValueError("contexts/residuals must be [N,D] with equal N")
        if x.shape[0] < 3 or x.shape[1] < 1 or r.shape[1] < 1:
            raise ValueError("confidence calibration requires at least three source rows")
        if not np.isfinite(x).all() or not np.isfinite(r).all():
            raise ValueError("confidence calibration data must be finite")
        for name, value in (
            ("residual_quantile", residual_quantile),
            ("support_quantile", support_quantile),
        ):
            if not np.isfinite(value) or not 0.5 <= value < 1.0:
                raise ValueError(f"{name} must be in [0.5,1)")
        if not np.isfinite(support_outer_multiplier) or support_outer_multiplier <= 1.0:
            raise ValueError("support_outer_multiplier must be > 1")

        context_mean = x.mean(axis=0)
        context_scale = x.std(axis=0)
        context_scale[context_scale < 1e-8] = 1.0
        x_z = (x - context_mean) / context_scale

        loo_distances = []
        for i in range(x_z.shape[0]):
            distances = np.linalg.norm(x_z - x_z[i], axis=1)
            distances[i] = np.inf
            loo_distances.append(float(distances.min()))
        support_radius = max(float(np.quantile(loo_distances, support_quantile)), 1e-6)

        residual_center = r.mean(axis=0)
        centered_norms = np.linalg.norm(r - residual_center, axis=1)
        residual_threshold = max(
            float(np.quantile(centered_norms, residual_quantile)), 1e-8
        )
        return cls(
            source_contexts_z=x_z,
            context_mean=context_mean,
            context_scale=context_scale,
            support_radius=support_radius,
            residual_center=residual_center,
            residual_threshold=residual_threshold,
            support_outer_multiplier=float(support_outer_multiplier),
        )

    @torch.no_grad()
    def confidence(
        self,
        raw_residual: torch.Tensor,
        context: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if raw_residual.ndim != 2 or raw_residual.shape[1] != self.residual_center.shape[0]:
            raise ValueError("raw_residual must be [B,visual_dim]")
        ctx = context.detach().cpu().numpy() if torch.is_tensor(context) else np.asarray(context)
        if ctx.ndim == 1:
            ctx = ctx[None, :]
        if ctx.ndim != 2 or ctx.shape[0] != raw_residual.shape[0]:
            raise ValueError("context batch size must match residual batch size")
        if ctx.shape[1] != self.context_mean.shape[0] or not np.isfinite(ctx).all():
            raise ValueError("context shape/non-finite values do not match source calibration")

        # torch.cdist is not implemented for every model dtype (notably some
        # BF16 paths). Keep confidence geometry in FP32, then cast the scalar
        # gate back to the visual residual dtype.
        device = raw_residual.device
        ctx_z = torch.as_tensor(
            (ctx - self.context_mean) / self.context_scale,
            device=device,
            dtype=torch.float32,
        )
        bank = torch.as_tensor(
            self.source_contexts_z, device=device, dtype=torch.float32
        )
        nearest = torch.cdist(ctx_z, bank).min(dim=1, keepdim=True).values
        inner = self.support_radius
        outer = self.support_radius * self.support_outer_multiplier
        support_conf = ((outer - nearest) / max(outer - inner, 1e-8)).clamp(0.0, 1.0)

        residual_center = torch.as_tensor(
            self.residual_center, device=device, dtype=torch.float32
        ).unsqueeze(0)
        residual_fp32 = raw_residual.float()
        residual_norm = torch.linalg.vector_norm(
            residual_fp32 - residual_center, dim=-1, keepdim=True
        )
        residual_conf = (
            1.0 - self.residual_threshold / residual_norm.clamp_min(1e-8)
        ).clamp(0.0, 1.0)
        return (support_conf * residual_conf).to(dtype=raw_residual.dtype)


def fit_source_calibration(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 10.0,
    content_rank: int = 2,
    folds: int = 5,
    seed: int = 0,
    standardize: bool = True,
) -> tuple[
    ConditionalSourceBaseline,
    SourceResidualProjector,
    SourceResidualConfidenceGate,
    dict,
]:
    """Fit source baseline, content filter and confidence gate from Clean data."""
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
    confidence_gate = SourceResidualConfidenceGate.fit(x, oof)
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
        "confidence_gate": {
            "residual_norm_threshold": confidence_gate.residual_threshold,
            "context_support_radius": confidence_gate.support_radius,
            "context_support_outer_radius": (
                confidence_gate.support_radius * confidence_gate.support_outer_multiplier
            ),
            "calibration_rows": int(x.shape[0]),
            "warning": "small source calibration set; log confidence and abstentions",
        },
    }
    return baseline, projector, confidence_gate, diagnostics


class TargetOnlyMeanEstimator:
    """Episode-local EMA of target residuals with optional source-only abstention."""

    def __init__(
        self,
        source_baseline: ConditionalSourceBaseline,
        momentum: float = 0.5,
        alpha: float = 1.0,
        clip_norm: float | None = None,
        residual_projector: SourceResidualProjector | None = None,
        confidence_gate: SourceResidualConfidenceGate | None = None,
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
        self.confidence_gate = confidence_gate
        self.ema: torch.Tensor | None = None
        self.last_raw_residual: torch.Tensor | None = None
        self.last_confidence: torch.Tensor | None = None

    def reset(self) -> None:
        self.ema = None
        self.last_raw_residual = None
        self.last_confidence = None

    @torch.no_grad()
    def raw_residual(
        self,
        visual: torch.Tensor,
        context: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
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
    def residual(self, visual, context) -> torch.Tensor:
        raw = self.raw_residual(visual, context)
        return raw if self.residual_projector is None else self.residual_projector.transform(raw)

    @torch.no_grad()
    def update(self, visual, context) -> torch.Tensor:
        raw = self.raw_residual(visual, context)
        residual = raw if self.residual_projector is None else self.residual_projector.transform(raw)
        if self.ema is None:
            self.ema = residual.clone()
        else:
            if self.ema.shape != residual.shape:
                raise ValueError("target batch shape changed; reset estimator before changing batch size")
            self.ema = self.momentum * self.ema + (1.0 - self.momentum) * residual
        self.last_raw_residual = raw
        if self.confidence_gate is None:
            self.last_confidence = torch.ones(
                (raw.shape[0], 1), device=raw.device, dtype=raw.dtype
            )
        else:
            self.last_confidence = self.confidence_gate.confidence(raw, context)
        return self.ema

    @torch.no_grad()
    def applied_correction(self) -> torch.Tensor:
        if self.ema is None:
            raise RuntimeError("update must be called before applied_correction")
        delta = self.ema.clone()
        if self.clip_norm is not None:
            norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            delta = delta * (self.clip_norm / norm).clamp(max=1.0)
        if self.last_confidence is not None:
            delta = delta * self.last_confidence
        return self.alpha * delta

    @torch.no_grad()
    def correction(self, visual, context) -> torch.Tensor:
        self.update(visual, context)
        return self.applied_correction()
