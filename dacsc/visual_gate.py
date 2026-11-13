"""Lightweight visual-token gate inserted between CoRP's DINO and Resampler.

Modes:
  identity    : return the exact input tensor (no arithmetic);
  suppression : gamma clamped to [0, 1];
  reweight    : gamma unrestricted (can amplify);
  signed      : gamma unrestricted (can be negative);
  scale       : backward-compatible alias of reweight.
"""

from __future__ import annotations

import types

import torch
from torch import nn

VALID_MODES = ("identity", "suppression", "reweight", "signed", "scale")


class VisualTokenGate(nn.Module):
    def __init__(self, view_names, dim: int, mode: str = "identity") -> None:
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        self.view_names = tuple(view_names)
        self.dim = int(dim)
        self.mode = mode
        self.gamma = nn.Parameter(torch.ones(len(self.view_names), self.dim))

    def _check(self, tokens: torch.Tensor, view: str) -> int:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("tokens must be a [B, N, D] tensor")
        if view not in self.view_names:
            raise KeyError(f"unknown view {view!r}")
        if tokens.shape[-1] != self.dim:
            raise ValueError(f"token dim {tokens.shape[-1]} != gate dim {self.dim}")
        return self.view_names.index(view)

    def forward(self, tokens: torch.Tensor, view: str) -> torch.Tensor:
        index = self._check(tokens, view)  # validate even for identity
        if self.mode == "identity":
            return tokens
        gamma = self.gamma[index].to(device=tokens.device, dtype=tokens.dtype)
        if self.mode == "suppression":
            gamma = gamma.clamp(0.0, 1.0)
        return tokens * gamma.view(1, 1, -1)

    @torch.no_grad()
    def set_uniform(self, value: float = 1.0) -> None:
        self.gamma.fill_(float(value))


class GatedRepresentationExtractor(nn.Module):
    """Wrap a frozen RepresentationExtractor and gate patch tokens before fusion."""

    def __init__(self, base, gate: VisualTokenGate) -> None:
        super().__init__()
        self.base = base
        self.gate = gate
        self.view_names = tuple(base.view_names)

    def _validate(self, images, state, instruction) -> int:
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
                raise ValueError(f"view {name} batch {images[name].shape[0]} != state batch {batch}")
        if isinstance(instruction, (list, tuple)) and len(instruction) != batch:
            raise ValueError("instruction count must match batch")
        return batch

    def forward(self, images, state, instruction):
        batch_size = self._validate(images, state, instruction)
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
