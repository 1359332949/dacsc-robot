"""Utilities for offline CoRP DINO patch diagnostics.

This module deliberately treats the patch grid and embedding width as runtime
properties. It contains no target-time adaptation code.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np


def as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy/PyTorch values to a finite-checkable CPU array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_patch_dict(
    patches: Mapping[str, Any],
    expected_views: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    """Validate per-view [B, N, C] patch tensors without assuming N or C."""
    if not isinstance(patches, Mapping) or not patches:
        raise ValueError("patches must be a non-empty per-view mapping")
    if expected_views is not None and tuple(patches) != tuple(expected_views):
        raise ValueError(
            f"view order mismatch: got {tuple(patches)}, expected {tuple(expected_views)}"
        )
    shapes: dict[str, list[int]] = {}
    batch_sizes = set()
    for view, value in patches.items():
        arr = as_numpy(value)
        if arr.ndim != 3:
            raise ValueError(f"{view}: expected [B,N,C], got shape {arr.shape}")
        if min(arr.shape) <= 0:
            raise ValueError(f"{view}: empty patch tensor {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{view}: patch tensor contains NaN/Inf")
        batch_sizes.add(int(arr.shape[0]))
        shapes[str(view)] = [int(x) for x in arr.shape]
    if len(batch_sizes) != 1:
        raise ValueError(f"per-view batch sizes differ: {sorted(batch_sizes)}")
    return shapes


def token_descriptor(tokens: Any, eps: float = 1e-6) -> np.ndarray:
    """Channel mean + log channel std over the token axis for one sample."""
    x = as_numpy(tokens).astype(np.float64, copy=False)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError(f"expected a single sample [1,N,C], got {x.shape}")
        x = x[0]
    if x.ndim != 2 or min(x.shape) <= 0:
        raise ValueError(f"expected [N,C] patch tokens, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("patch tokens contain NaN/Inf")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return np.concatenate([mean, np.log(std + eps)]).astype(np.float64)


def cosine_self_similarity(tokens: Any, eps: float = 1e-12) -> np.ndarray:
    """Return the actual N x N cosine self-similarity matrix."""
    x = as_numpy(tokens).astype(np.float64, copy=False)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError(f"expected a single sample [1,N,C], got {x.shape}")
        x = x[0]
    if x.ndim != 2 or min(x.shape) <= 0:
        raise ValueError(f"expected [N,C] patch tokens, got {x.shape}")
    denom = np.linalg.norm(x, axis=1, keepdims=True).clip(min=eps)
    unit = x / denom
    return unit @ unit.T


def normalized_distance(a: Any, b: Any, scale: Any | None = None) -> float:
    """RMS Euclidean distance; optional scale is estimated on Source Fit only."""
    x = as_numpy(a).astype(np.float64, copy=False).reshape(-1)
    y = as_numpy(b).astype(np.float64, copy=False).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"descriptor shape mismatch: {x.shape} vs {y.shape}")
    delta = x - y
    if scale is not None:
        s = as_numpy(scale).astype(np.float64, copy=False).reshape(-1)
        if s.shape != delta.shape:
            raise ValueError(f"scale shape mismatch: {s.shape} vs {delta.shape}")
        delta = delta / np.maximum(s, 1e-6)
    return float(np.sqrt(np.mean(delta * delta)))


def random_clean_donor(
    records: Sequence[Mapping[str, Any]],
    index: int,
    seed: int,
) -> int | None:
    """Choose a different-state Clean donor within the same role/window/chunk."""
    target = records[index]
    key = (target.get("role"), target.get("window"), target.get("chunk_index"))
    candidates = [
        i for i, record in enumerate(records)
        if i != index
        and (record.get("role"), record.get("window"), record.get("chunk_index")) == key
        and int(record.get("state_id", -1)) != int(target.get("state_id", -1))
        and bool(record.get("pairing_pass", False))
    ]
    if not candidates:
        return None
    rng = np.random.default_rng(int(seed) + 104729 * int(index))
    return int(candidates[int(rng.integers(0, len(candidates)))])


def patch_distance_summary(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    """Per-view RMS token displacement, with no cross-view averaging."""
    if tuple(a) != tuple(b):
        raise ValueError("patch view sets/order differ")
    out: dict[str, float] = {}
    for view in a:
        x, y = as_numpy(a[view]), as_numpy(b[view])
        if x.shape != y.shape:
            raise ValueError(f"{view}: patch shape mismatch {x.shape} vs {y.shape}")
        out[str(view)] = float(np.sqrt(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2)))
    return out



def state_balanced_mean(
    records: Sequence[Mapping[str, Any]],
    metric_key: str,
    nested_key: str | None = None,
) -> dict[str, Any] | None:
    """Average numeric metric dictionaries within state, then states equally.

    Each record must resolve to a mapping of scalar numeric metrics. This
    deliberately prevents treating multiple chunks from one state as
    independent statistical units.
    """
    by_state: dict[int, list[dict[str, float]]] = {}
    for record in records:
        value = record.get(metric_key)
        if value is None:
            continue
        if nested_key is not None:
            if not isinstance(value, Mapping):
                raise TypeError(f"{metric_key} must be a mapping for nested lookup")
            value = value.get(nested_key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise TypeError("state_balanced_mean expects a numeric metric mapping")
        metrics = {str(k): float(v) for k, v in value.items()}
        if not metrics or not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("metric mappings must be non-empty and finite")
        by_state.setdefault(int(record["state_id"]), []).append(metrics)

    if not by_state:
        return None
    metric_names = tuple(next(iter(by_state.values()))[0].keys())
    for items in by_state.values():
        if any(tuple(item.keys()) != metric_names for item in items):
            raise ValueError("metric keys/order differ across records")

    state_means = {
        state: {
            key: float(np.mean([item[key] for item in items]))
            for key in metric_names
        }
        for state, items in by_state.items()
    }
    return {
        "n_states": len(state_means),
        "n_records": sum(len(items) for items in by_state.values()),
        "state_balanced_mean": {
            key: float(np.mean([metrics[key] for metrics in state_means.values()]))
            for key in metric_names
        },
        "per_state": state_means,
    }
