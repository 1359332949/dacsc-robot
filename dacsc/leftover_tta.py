"""Interface-level leftover-consistency test-time adaptation (pilot implementation).

Correct model location (verified against corp/action_generator.py):
  * `block.encode_prefix(...)` also calls `attention.attend`, but its tokens are the
    representation tokens (T_rep ~ 90+), so that call must NEVER be modified.
  * `block.forward(...)` calls `attention.attend` once per Euler step with the
    action tokens (T = action_horizon = 12). The Euler loop is `num_sample_steps`
    calls after the single prefix call.

Hook indexing:
    call index 0            -> prefix encoding (skip)
    call index 1..num_steps -> Euler step 1..num_steps (action-token write)

Adaptation objective (test time, no labels):
    chunk' = sample with the interface transform active
    target = previous chunk tail (actions predicted earlier, not yet executed)
    loss   = MSE(chunk'[:H-E], target[E:]) + reg * ||U V^T||^2
Only U,V (low-rank, D x r) are optimised; everything else is frozen.
"""

from __future__ import annotations

import torch
from torch import nn

from corp.action_generator import DiTActionGenerator


class LowRankInterfaceTransform:
    def __init__(self, attention, hidden_dim: int, rank: int = 8, reg: float = 1e-3,
                 action_basis: torch.Tensor | None = None,
                 num_steps: int | None = None, last_step_only: bool = False):
        self.attention = attention
        self.original = attention.attend
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.reg = reg
        device = next(attention.parameters()).device
        # LoRA-style initialisation: one factor random, the other zero. With BOTH
        # factors zero the product U V^T is a saddle point and gradients vanish.
        self.u = nn.Parameter(torch.zeros(hidden_dim, rank, device=device))
        self.v = nn.Parameter(torch.randn(hidden_dim, rank, device=device) * 1e-3)
        self.optimizer = torch.optim.Adam([self.u, self.v], lr=1e-3, weight_decay=0.0)
        # optional protection: remove the action-sensitive component from the
        # update, so the policy's own compensation (measured to be predictive of
        # success) is preserved. action_basis: [D, r_a] orthonormal.
        self.action_basis = action_basis.to(device) if action_basis is not None else None
        self.update_a_frac = 0.0
        self.loss_trace: list[tuple[float, float, float]] = []
        self.adapted_once = False
        self.gated = False
        self.call = 0
        self.active = False
        self.num_steps = int(num_steps) if num_steps is not None else None
        self.last_step_only = bool(last_step_only)
        attention.attend = self._attend  # type: ignore[assignment]

    def _apply(self, output: torch.Tensor) -> torch.Tensor:
        # z' = z + U V^T z   (low-rank, reversible for small U V^T)
        correction = (output @ self.v) @ self.u.T
        if self.action_basis is not None:
            flat = correction.reshape(-1, correction.shape[-1])
            proj = (flat @ self.action_basis) @ self.action_basis.T
            if correction.norm() > 0:
                self.update_a_frac = float(proj.norm() / correction.norm())
            correction = (flat - proj).reshape(correction.shape)
        return output + correction

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        index = self.call
        self.call += 1
        if index == 0:
            # prefix encoding over representation tokens: never touch
            return output
        if self.active:
            if self.last_step_only and self.num_steps is not None and index < self.num_steps:
                return output
            return self._apply(output)
        return output

    def reset(self):
        self.call = 0

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


def sample_with_grad(generator: DiTActionGenerator, representation, num_steps: int | None = None) -> torch.Tensor:
    """Same Euler recursion as DiTActionGenerator.sample, but without @torch.no_grad."""
    steps = generator.num_sample_steps if num_steps is None else num_steps
    action = torch.randn(
        (representation.tokens.shape[0], generator.action_horizon, generator.action_dim),
        device=representation.tokens.device,
        dtype=representation.action_dtype,
    )
    prefix = generator.prepare_prefix(representation)
    for step in range(steps):
        tau = torch.full(
            (action.shape[0],),
            step / float(steps),
            device=action.device,
            dtype=action.dtype,
        )
        action = action + generator(action, tau, prefix) * (1.0 / float(steps))
    return action


def leftover_loss(chunk: torch.Tensor, target: torch.Tensor, exec_len: int) -> torch.Tensor:
    """chunk [H, A] (new), target [H-E, A] (previous chunk tail at the same absolute times)."""
    horizon = chunk.shape[0]
    overlap = horizon - exec_len
    if overlap <= 0:
        return torch.zeros((), device=chunk.device)
    return torch.nn.functional.mse_loss(chunk[:overlap], target[:overlap])
