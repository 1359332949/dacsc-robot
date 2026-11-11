"""SALT-style test-time adaptation for CoRP: bounded multiplicative singular-value
scaling of selected attention projections, updated only at test time.

Design (follows the SALT description: frozen backbone, selected attention
projections updated with bounded multiplicative singular-value scaling, Adam,
few steps, leftover action-chunk self-supervision):

    W = U S V^T                    (each selected Linear weight)
    s(phi) = 1 + bound * tanh(phi) (learnable multiplicative factors)
    W' = U diag(S * s(phi)) V^T    (init phi=0 -> W'=W exactly)
    y  = x W'^T + b

Only `phi` parameters receive gradients; all CoRP parameters stay frozen.
The caller must reset() per episode.
"""

from __future__ import annotations

import torch
from torch import nn


class SaltProjectionAdapter:
    def __init__(self, module: nn.Linear, bound: float = 0.1, full_rank: bool = False):
        self.module = module
        self.bound = float(bound)
        w = module.weight.detach().float()                      # [out, in]
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        rank = s.numel()
        self.u = u.contiguous()
        self.s = s.contiguous()
        self.v = vh.contiguous()                                # [r, in]
        self.phi = nn.Parameter(torch.zeros(rank, device=w.device))
        self._orig_forward = module.forward
        self._w0 = module.weight.detach().clone()               # exact identity at phi=0
        module.forward = self._forward                          # type: ignore[assignment]

    def effective_weight(self, dtype=None) -> torch.Tensor:
        if float(self.phi.detach().abs().max()) == 0.0:
            return self._w0                                      # bit-exact Frozen weight
        scale = 1.0 + self.bound * torch.tanh(self.phi)
        w = (self.u * (self.s * scale)) @ self.v
        return w.to(self.module.weight.dtype)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight()
        return nn.functional.linear(x, w, self.module.bias)

    def reset(self) -> None:
        with torch.no_grad():
            self.phi.zero_()

    def close(self) -> None:
        self.module.forward = self._orig_forward  # type: ignore[assignment]


class SaltTTA:
    """Owner of several SaltProjectionAdapter instances + optimizer."""

    def __init__(self, policy, target_modules: list[str], bound: float = 0.1, lr: float = 1e-3):
        self.policy = policy
        blocks = policy.action_generator.blocks
        selected = []
        for name in target_modules:
            attn = blocks[0].attention
            selected.append(getattr(attn, name))
        for p in policy.parameters():
            p.requires_grad_(False)
        self.adapters = [SaltProjectionAdapter(m, bound=bound) for m in selected]
        self.params = [a.phi for a in self.adapters]
        self.optimizer = torch.optim.Adam(self.params, lr=lr)

    def reset(self) -> None:
        for a in self.adapters:
            a.reset()
        self.optimizer = torch.optim.Adam(self.params, lr=self.optimizer.param_groups[0]["lr"])

    def close(self) -> None:
        for a in self.adapters:
            a.close()

    def grad_norm(self) -> float:
        return float(sum(float(p.grad.norm()) ** 2 for p in self.params if p.grad is not None) ** 0.5)
