"""Lightweight visual-token gate inserted between CoRP's DINO and Resampler.

The gate operates on each view's [B, N, D] patch-token tensor.  In "identity"
mode it returns the exact input tensor (no multiply-by-one), so a no-op gate can
be audited end-to-end against the unmodified representation extractor.
"""

from __future__ import annotations

import types

import torch
from torch import nn


class VisualTokenGate(nn.Module):
    def __init__(self, view_names, dim: int, mode: str = "identity") -> None:
        super().__init__()
        if mode not in ("identity", "scale"):
            raise ValueError("mode must be 'identity' or 'scale'")
        self.view_names = tuple(view_names)
        self.dim = int(dim)
        self.mode = mode
        # One scalar per (view, embedding dimension); initialized to 1.0.
        self.gamma = nn.Parameter(torch.ones(len(self.view_names), self.dim))

    def forward(self, tokens: torch.Tensor, view: str) -> torch.Tensor:
        if self.mode == "identity":
            return tokens
        if view not in self.view_names:
            raise KeyError(f"unknown view {view!r}")
        index = self.view_names.index(view)
        gamma = self.gamma[index].to(device=tokens.device, dtype=tokens.dtype)
        return tokens * gamma.view(1, 1, -1)

    @torch.no_grad()
    def set_uniform(self, value: float = 1.0) -> None:
        self.gamma.fill_(float(value))


class GatedRepresentationExtractor(nn.Module):
    """Wrap a frozen RepresentationExtractor and gate patch tokens before fusion.

    The wrapper reuses the base extractor's modules without modifying the
    checkpoint.  It is intended for V0 interface checks and later V2 gates.
    """

    def __init__(self, base, gate: VisualTokenGate) -> None:
        super().__init__()
        self.base = base
        self.gate = gate
        self.view_names = tuple(base.view_names)

    def forward(self, images, state, instruction):
        batch_size = state.shape[0]
        patches = self.base.visual_encoder(
            torch.cat([images[name] for name in self.view_names], dim=0)
        )
        patch_tokens = {
            name: patches[index * batch_size : (index + 1) * batch_size]
            for index, name in enumerate(self.view_names)
        }
        patch_tokens = {
            name: self.gate(tokens, name) for name, tokens in patch_tokens.items()
        }
        language_tokens, pooled_language = self.base.language_encoder(instruction)
        if language_tokens.shape[0] != batch_size:
            raise ValueError("instruction count must match the observation batch size")
        visual_tokens = self.base.cross_attention_resampler(
            patch_tokens, pooled_language
        )
        visual_language = torch.cat(
            [visual_tokens, language_tokens.to(visual_tokens.dtype)], dim=1
        )
        state_token = self.base.state_encoder(state)
        return types.SimpleNamespace(
            tokens=torch.cat([visual_language, state_token], dim=1),
            pooled_language=pooled_language,
            action_dtype=visual_language.dtype,
        )
