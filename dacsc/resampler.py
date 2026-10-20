"""Learned-query token resampler for DINOv2 patch tokens.

This is a compact CoRP-style resampler: a small set of learned queries
cross-attend to the patch tokens, then each query is projected and the
query features are concatenated into one per-view vector.

It is deliberately kept small so the policy remains a controlled baseline:
DINOv2 is shared across views; the resampler only compresses 256 patch
tokens to K task-adaptive tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LearnedQueryResampler(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64,
                 num_queries: int = 48, n_heads: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        assert in_dim % n_heads == 0, (in_dim, n_heads)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_queries = int(num_queries)
        self.query = nn.Parameter(torch.randn(1, self.num_queries, in_dim) * 0.02)
        self.q_norm = nn.LayerNorm(in_dim)
        self.kv_norm = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(
            in_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 4 * in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * in_dim, in_dim),
        )
        self.out_proj = nn.Linear(in_dim, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (N,T,D) -> (N,K*out_dim)."""
        n = tokens.shape[0]
        q = self.query.expand(n, -1, -1)
        kv = self.kv_norm(tokens)
        z, _ = self.attn(self.q_norm(q), kv, kv, need_weights=False)
        z = z + self.ffn(z)
        z = self.out_proj(z)
        return z.reshape(n, -1)
