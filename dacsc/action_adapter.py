"""Zero-initialized residual adapter for frozen CoRP visual tokens."""

from __future__ import annotations
import torch
from torch import nn


class ResidualVisualAdapter(nn.Module):
    """Small bottleneck residual adapter; output starts at zero."""

    def __init__(self, dim: int = 768, rank: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.act = nn.SiLU()
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be [B,T,D], got {tuple(tokens.shape)}")
        return tokens + self.up(self.act(self.down(self.norm(tokens))))
