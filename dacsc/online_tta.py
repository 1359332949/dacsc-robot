"""Causal online update of the visual adapter only."""

from __future__ import annotations
import copy
import math
import torch


class OnlineTTAUpdater:
    def __init__(self, adapter, predictor, encode_fn, lr: float = 1e-4,
                 max_grad_norm: float = 1.0, weight_decay: float = 1e-4):
        self.adapter = adapter
        self.predictor = predictor
        self.encode_fn = encode_fn  # differentiable patch -> pooled z
        self.optimizer = torch.optim.Adam(adapter.parameters(), lr=lr, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self._initial_state = copy.deepcopy(adapter.state_dict())

    def reset_episode(self) -> None:
        self.adapter.load_state_dict(self._initial_state)
        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.optimizer.param_groups[0]["lr"],
                                          weight_decay=self.optimizer.param_groups[0].get("weight_decay", 0.0))

    def update(self, patch_tokens, context, z_next_observed, state_t, actions_exec, lang_embed=None):
        """Run one update after an action chunk has actually executed.

        patch_tokens: dict[view -> [B,N,D]] current target-domain tokens.
        z_next_observed: pooled representation from the actually observed next
        target-domain frame.  It is detached before entering the loss.
        """
        self.predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad_(False)
        self.adapter.train()
        self.optimizer.zero_grad(set_to_none=True)
        adapted = {v: self.adapter(t, v, context) for v, t in patch_tokens.items()}
        z_t = self.encode_fn(adapted)
        z_next_observed = z_next_observed.detach()
        pred = self.predictor(z_t, state_t, actions_exec, lang_embed)
        loss = torch.nn.functional.mse_loss(pred, z_next_observed)
        if not torch.isfinite(loss):
            return {"updated": False, "reason": "non_finite_loss", "loss": None}
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.max_grad_norm)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            return {"updated": False, "reason": "non_finite_grad", "loss": float(loss.detach())}
        self.optimizer.step()
        changed = any(torch.isfinite(p).all() for p in self.adapter.parameters()) and grad_norm > 0
        return {"updated": bool(changed), "loss": float(loss.detach()),
                "grad_norm": float(grad_norm), "predictor_frozen": True}
