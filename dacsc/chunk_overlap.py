"""Predictor-free target-only baseline: action-chunk overlap consistency.

This is deliberately a baseline, not the final method.  It updates only the
visual adapter using the overlap between an earlier frozen chunk and a newly
generated chunk after K actions have actually been executed.  It does not use
Clean observations, success labels, or a trained predictor.
"""

from __future__ import annotations
import copy
import torch
from corp.preprocessing import prepare_observation


def sample_chunk(extractor, policy, observation, z0, context=None):
    """Differentiable Euler sampling through the frozen action generator."""
    batch = prepare_observation(
        observation, config=policy.config, normalizer=policy.normalizer,
        device=next(policy.parameters()).device)
    representation = extractor(batch["images"], batch["state"], batch["instruction"], context=context)
    gen = policy.action_generator
    action = z0.to(device=representation.tokens.device, dtype=representation.action_dtype).clone()
    prefix = gen.prepare_prefix(representation)
    for step in range(gen.num_sample_steps):
        tau = torch.full((action.shape[0],), step / float(gen.num_sample_steps),
                         device=action.device, dtype=action.dtype)
        action = action + gen(action, tau, prefix) * (1.0 / float(gen.num_sample_steps))
    return action


class ChunkOverlapUpdater:
    def __init__(self, policy, extractor, adapter, lr: float = 1e-5,
                 anchor_weight: float = 1e-4, max_grad_norm: float = 1.0):
        self.policy = policy
        self.extractor = extractor
        self.adapter = adapter
        self.optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
        self.anchor_weight = anchor_weight
        self.max_grad_norm = max_grad_norm
        self.reference_chunk = None
        self.reference_z0 = None
        self.reference_context = None
        self.initial_params = {k: v.detach().clone() for k, v in adapter.state_dict().items()}

    @torch.no_grad()
    def set_reference(self, observation, z0, context=None):
        self.reference_chunk = sample_chunk(self.extractor, self.policy, observation, z0, context).detach()
        self.reference_z0 = z0.detach().clone()
        self.reference_context = None if context is None else context.detach().clone()

    def update(self, observation_after, executed_k: int, context=None, z0=None):
        if self.reference_chunk is None:
            raise RuntimeError("set_reference must be called first")
        if z0 is None:
            z0 = self.reference_z0
        if context is None:
            context = self.reference_context
        horizon = self.reference_chunk.shape[1]
        if not (0 < executed_k < horizon):
            raise ValueError("executed_k must be in (0, horizon)")
        before = {k: v.detach().clone() for k, v in self.adapter.state_dict().items()}
        self.optimizer.zero_grad(set_to_none=True)
        new_chunk = sample_chunk(self.extractor, self.policy, observation_after, z0, context)
        old_suffix = self.reference_chunk[:, executed_k:, :]
        new_prefix = new_chunk[:, :horizon - executed_k, :]
        consistency = torch.nn.functional.mse_loss(new_prefix, old_suffix.detach())
        # Keep the adapted representation close to the input tokens.
        # The extractor's adapter output is not exposed here; use parameter
        # distance as a cheap anchor proxy in this baseline.
        anchor = sum((p - self.initial_params[k]).pow(2).sum()
                     for k, p in self.adapter.named_parameters())
        loss = consistency + self.anchor_weight * anchor
        if not torch.isfinite(loss):
            return {"updated": False, "reason": "non_finite_loss", "loss": None}
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.max_grad_norm)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            return {"updated": False, "reason": "non_finite_grad", "loss": float(loss.detach())}
        self.optimizer.step()
        all_finite = all(bool(torch.isfinite(p).all()) for p in self.adapter.parameters())
        changed = any(not torch.equal(p.detach(), before[k])
                      for k, p in self.adapter.named_parameters())
        if not all_finite:
            self.adapter.load_state_dict(before)
            self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.optimizer.param_groups[0]["lr"])
            return {"updated": False, "reason": "non_finite_param", "loss": float(loss.detach())}
        return {"updated": bool(changed), "loss": float(loss.detach()),
                "consistency": float(consistency.detach()), "grad_norm": float(grad_norm),
                "param_delta": sum(float((p.detach() - before[k]).abs().sum())
                                   for k, p in self.adapter.named_parameters())}
