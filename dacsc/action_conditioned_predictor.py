"""Frozen action-conditioned predictor used as the TTA self-supervision target."""

from __future__ import annotations
import torch
from torch import nn


class ActionConditionedPredictor(nn.Module):
    """Predict the next pooled visual representation from current state/action.

    Deployment inputs are strictly causal: z_t, q_t, instruction embedding and
    the actually executed action chunk.  The next observation is only used as
    the target after execution.
    """

    def __init__(self, z_dim: int = 768, state_dim: int = 8, action_dim: int = 7,
                 horizon: int = 12, lang_dim: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = z_dim + state_dim + action_dim * horizon + lang_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, z_dim))
        self.z_dim = z_dim; self.state_dim = state_dim
        self.action_dim = action_dim; self.horizon = horizon; self.lang_dim = lang_dim

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, z_t, state_t, actions_exec, lang_embed=None):
        if z_t.ndim != 2 or z_t.shape[-1] != self.z_dim:
            raise ValueError(f"z_t must be [B,{self.z_dim}]")
        if state_t.shape != (z_t.shape[0], self.state_dim):
            raise ValueError("state shape mismatch")
        if actions_exec.shape != (z_t.shape[0], self.horizon, self.action_dim):
            raise ValueError("action shape mismatch")
        if lang_embed is None:
            lang_embed = z_t.new_zeros(z_t.shape[0], self.lang_dim)
        if lang_embed.ndim != 2 or lang_embed.shape[-1] != self.lang_dim:
            raise ValueError("language embedding shape mismatch")
        x = torch.cat([z_t, state_t, actions_exec.reshape(z_t.shape[0], -1), lang_embed], dim=-1)
        return self.net(x)
