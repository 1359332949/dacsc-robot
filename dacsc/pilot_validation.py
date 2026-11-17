"""Fail-closed validation helpers for the closed-loop pilot."""

from __future__ import annotations

ALLOWED_METHODS = {"frozen", "first_observation", "paired_first", "online", "style_norm", "style_mean", "leftover_select", "frozen_predict"}


def validate_methods(raw: str) -> list[str]:
    methods = [m.strip() for m in raw.split(",") if m.strip()]
    if not methods:
        raise ValueError("at least one method is required")
    if len(methods) != len(set(methods)):
        raise ValueError(f"duplicate methods: {methods}")
    unknown = set(methods) - ALLOWED_METHODS
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    return methods


def parse_init_indices(raw: str) -> list[int]:
    try:
        values = [int(i) for i in raw.split(",") if i.strip()]
    except ValueError as exc:
        raise ValueError(f"init indices must be integers: {raw}") from exc
    if not values or len(values) != len(set(values)):
        raise ValueError("init indices must be non-empty and unique")
    return values


def validate_executed_k(value: int, horizon: int) -> int:
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"invalid action horizon: {horizon}")
    if not 1 <= value <= horizon:
        raise ValueError(f"executed-k must be in [1,{horizon}], got {value}")
    return value


def validate_init_indices(indices: list[int], n_states: int,
                           heldout: set[int], fit_ids: list[int],
                           require_heldout: bool = True) -> None:
    if len(indices) != len(set(indices)):
        raise ValueError("init indices must be unique")
    for index in indices:
        if index < 0 or index >= n_states:
            raise ValueError(f"init index out of range 0..{n_states - 1}: {index}")
    if require_heldout:
        missing = set(indices) - set(heldout)
        if missing:
            raise ValueError(f"init indices must be heldout: {sorted(missing)}")
    overlap = set(indices) & set(fit_ids)
    if overlap:
        raise ValueError(f"init indices overlap source fit IDs: {sorted(overlap)}")
