"""Low-capacity residual adapter between CoRP DINO tokens and Resampler.

The adapter starts as an exact identity (zero-initialized output layer) and is
the only trainable module in the TTA path.
"""

from __future__ import annotations
import torch
from torch import nn


class ResidualVisualAdapter(nn.Module):
    def __init__(self, view_names, dim: int = 384, context_dim: int = 0,
                 hidden_dim: int = 128, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.view_names = tuple(view_names)
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.residual_scale = float(residual_scale)
        in_dim = self.dim + self.context_dim
        self.norm = nn.LayerNorm(self.dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, self.dim)
        self.view_embedding = nn.Parameter(torch.zeros(len(self.view_names), hidden_dim))
        self._init_identity()

    def _init_identity(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.view_embedding)

    def set_identity(self) -> None:
        self._init_identity()

    def forward(self, tokens: torch.Tensor, view: str, context: torch.Tensor | None = None) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("tokens must be [B, N, D]")
        if tokens.shape[-1] != self.dim:
            raise ValueError(f"token dim {tokens.shape[-1]} != adapter dim {self.dim}")
        if view not in self.view_names:
            raise KeyError(f"unknown view {view!r}")
        batch = tokens.shape[0]
        if self.context_dim:
            if context is None:
                context = tokens.new_zeros(batch, self.context_dim)
            if context.shape != (batch, self.context_dim):
                raise ValueError(f"context must be [B,{self.context_dim}], got {tuple(context.shape)}")
        features = self.norm(tokens)
        if self.context_dim:
            ctx = context.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(1).expand(-1, tokens.shape[1], -1)
            features = torch.cat([features, ctx], dim=-1)
        hidden = self.act(self.fc1(features))
        hidden = hidden + self.view_embedding[self.view_names.index(view)].to(hidden.dtype)
        residual = self.fc2(hidden) * self.residual_scale
        return tokens + residual


class AdaptedRepresentationExtractor(nn.Module):
    """Wrap a frozen CoRP RepresentationExtractor with the residual adapter."""

    def __init__(self, base, adapter: ResidualVisualAdapter) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.view_names = tuple(base.view_names)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, images, state, instruction, context: torch.Tensor | None = None):
        import types
        if not isinstance(images, dict):
            raise TypeError("images must be a dict")
        missing = [v for v in self.view_names if v not in images]
        if missing:
            raise KeyError(f"missing views: {missing}")
        if state.ndim != 2:
            raise ValueError("state must be [B, state_dim]")
        batch = state.shape[0]
        for name in self.view_names:
            if images[name].shape[0] != batch:
                raise ValueError(f"view {name} batch mismatch")
        patches = self.base.visual_encoder(
            torch.cat([images[name] for name in self.view_names], dim=0))
        patch_tokens = {
            name: patches[i * batch:(i + 1) * batch]
            for i, name in enumerate(self.view_names)
        }
        patch_tokens = {
            name: self.adapter(tokens, name, context)
            for name, tokens in patch_tokens.items()
        }
        language_tokens, pooled_language = self.base.language_encoder(instruction)
        visual_tokens = self.base.cross_attention_resampler(patch_tokens, pooled_language)
        visual_language = torch.cat(
            [visual_tokens, language_tokens.to(visual_tokens.dtype)], dim=1)
        state_token = self.base.state_encoder(state)
        return types.SimpleNamespace(
            tokens=torch.cat([visual_language, state_token], dim=1),
            pooled_language=pooled_language,
            action_dtype=visual_language.dtype,
        )
