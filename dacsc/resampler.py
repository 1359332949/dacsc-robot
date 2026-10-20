"""Learned-query token resampler for DINOv2 patch tokens.

The output is a token sequence ``(N, K, out_dim)``; it is not flattened.
Language conditioning is applied as FiLM to the learned queries before
cross-attention, matching the intended CoRP-style interface.

This is still a compact implementation: queries are shared across camera
views, while each view has its own patch-token key/value sequence.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LearnedQueryResampler(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 64,
                 num_queries: int = 48, n_heads: int = 6,
                 dropout: float = 0.0, lang_dim: int | None = None):
        super().__init__()
        assert in_dim % n_heads == 0, (in_dim, n_heads)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_queries = int(num_queries)
        self.lang_dim = int(lang_dim) if lang_dim is not None else None
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
        if self.lang_dim is not None:
            self.lang_film = nn.Sequential(
                nn.LayerNorm(self.lang_dim),
                nn.Linear(self.lang_dim, 2 * in_dim),
            )

    def forward(self, tokens: torch.Tensor, lang_emb: torch.Tensor | None = None):
        """tokens: (N,T,D); lang_emb: (N,lang_dim) -> (N,K,out_dim)."""
        n = tokens.shape[0]
        q = self.query.expand(n, -1, -1)
        if self.lang_dim is not None and lang_emb is not None:
            gamma_beta = self.lang_film(lang_emb)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            q = q * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        kv = self.kv_norm(tokens)
        z, _ = self.attn(self.q_norm(q), kv, kv, need_weights=False)
        z = z + self.ffn(z)
        return self.out_proj(z)
