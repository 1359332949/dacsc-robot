"""Deployable runtime failure monitor for frozen VLA rollouts.

Inputs are limited to observations available at deployment: proprioception,
executed actions, and frozen visual features. No simulator object/contact truth.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class CausalFailureMonitor(nn.Module):
    def __init__(self, state_dim: int = 8, action_dim: int = 7,
                 visual_dim: int = 768, visual_proj: int = 32,
                 hidden_dim: int = 128, num_layers: int = 2, window_size: int = 10):
        super().__init__()
        self.window_size = window_size
        self.visual_proj = nn.Linear(visual_dim, visual_proj)
        self.temporal = nn.GRU(
            input_size=state_dim + action_dim + visual_proj,
            hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, state_seq, action_seq, visual_seq):
        # shapes: [B,W,8], [B,W,7], [B,W,768]
        if state_seq.shape[1] != self.window_size:
            raise ValueError(f"window mismatch: {state_seq.shape[1]} != {self.window_size}")
        v = self.visual_proj(visual_seq)
        x = torch.cat([state_seq, action_seq, v], dim=-1)
        out, _ = self.temporal(x)
        return self.head(out[:, -1, :]).squeeze(-1)
