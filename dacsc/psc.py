"""PSC-style test-time source-support projection (arXiv:2603.09338).

The method is label-free and training-free: source clean features are summarised
by their mean mu and top-r principal subspace P; at test time the cross-attention
write is replaced by

    w' = mu + P (w - mu)

i.e. the component of the write that lies outside the source support is dropped
(PSC's "source predictive support alignment"; the residual spectral calibration
of the paper is NOT applied, because it requires target statistics and our V12/V13
measurements show it does not help here).

This module is deliberately tiny so that the evaluation script can install it for
the duration of one model call.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


class SourceProjector:
    """Project the cross-attention write onto the source support.

    beta scales how much of the out-of-support residual is kept:
    beta=0 -> pure projection (PSC/V17 default), beta=1 -> identity.
    """

    def __init__(
        self,
        attention_module,
        mean: torch.Tensor,
        basis: torch.Tensor,
        beta: float = 0.0,
    ):
        if mean.ndim != 1 or basis.ndim != 2 or basis.shape[1] != mean.shape[0]:
            raise ValueError("PSC stats must be mean [D] and basis [r, D]")
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.attention = attention_module
        self.original = attention_module.attend
        self.mean = mean
        self.basis = basis
        self.beta = float(beta)
        attention_module.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        mean = self.mean.to(device=output.device, dtype=output.dtype)
        basis = self.basis.to(device=output.device, dtype=output.dtype)
        centred = output - mean
        inside = (centred @ basis.T) @ basis
        outside = centred - inside
        return mean + inside + self.beta * outside

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


def load_stats(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    return {
        "mean": torch.tensor(payload["mean"], dtype=torch.float32),
        "basis": torch.tensor(payload["basis"], dtype=torch.float32),
        "rank": int(payload["rank"]),
        "task": payload.get("task"),
    }


def install(model, stats: dict, beta: float = 0.0) -> SourceProjector:
    attention = model.action_generator.blocks[0].attention
    return SourceProjector(attention, stats["mean"], stats["basis"], beta=beta)
