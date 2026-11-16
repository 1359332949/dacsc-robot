"""Predictor-free target-only baseline: action-chunk overlap consistency.

This is deliberately a baseline, not the final method.  It updates only the
visual adapter using the overlap between an actually executed chunk and a newly
generated chunk after K actions.  It does not use Clean observations, success
labels, or a trained predictor.
"""

from __future__ import annotations
import torch
from corp.preprocessing import prepare_observation


def sample_chunk(extractor, policy, observation, z0, context=None, return_normalized=False):
    """Differentiable Euler sampling through the frozen action generator.

    Returns actions in LIBERO controller units unless return_normalized=True.
    The graph is kept so an adapter update can differentiate through the
    frozen generator.
    """
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
    if return_normalized:
        return action
    return policy.normalizer.unnormalize_action(action)


class ChunkOverlapUpdater:
    def __init__(self, policy, extractor, adapter, lr: float = 1e-5,
                 parameter_anchor_weight: float = 1e-4, max_grad_norm: float = 1.0):
        self.policy = policy
        self.extractor = extractor
        self.adapter = adapter
        self.optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
        self.parameter_anchor_weight = parameter_anchor_weight
        self.max_grad_norm = max_grad_norm
        self.initial_params = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
        self.reference_normalized = None
        self.reference_z0 = None
        self.reference_context = None

    def reset_episode(self) -> None:
        self.adapter.load_state_dict(self.initial_params)
        self.optimizer = torch.optim.Adam(self.adapter.parameters(),
                                          lr=self.optimizer.param_groups[0]["lr"])
        self.reference_normalized = None
        self.reference_z0 = None
        self.reference_context = None

    @torch.no_grad()
    def set_reference(self, observation, z0, context=None):
        """Generate the chunk that the caller must actually execute.

        Returns normalized reference (used for overlap loss) and unnormalized
        controller actions (the exact chunk to send to the environment).
        """
        ref_norm = sample_chunk(self.extractor, self.policy, observation, z0,
                                context, return_normalized=True).detach()
        self.reference_normalized = ref_norm
        self.reference_z0 = z0.detach().clone()
        self.reference_context = None if context is None else context.detach().clone()
        ref_exec = self.policy.normalizer.unnormalize_action(ref_norm)
        return {"normalized": ref_norm, "executed": ref_exec}

    @torch.no_grad()
    def _loss_only(self, observation_after, z0, context, executed_k):
        chunk = sample_chunk(self.extractor, self.policy, observation_after, z0,
                             context, return_normalized=True)
        h = self.reference_normalized.shape[1]
        return torch.nn.functional.mse_loss(
            chunk[:, :h - executed_k], self.reference_normalized[:, executed_k:])

    def update(self, observation_after, executed_k: int, context=None, z0=None):
        if self.reference_normalized is None:
            raise RuntimeError("set_reference must be called first")
        if z0 is None:
            z0 = self.reference_z0
        if context is None:
            context = self.reference_context
        h = self.reference_normalized.shape[1]
        if not (0 < executed_k < h):
            raise ValueError("executed_k must be in (0, horizon)")
        loss_before = float(self._loss_only(observation_after, z0, context, executed_k))
        before = {k: v.detach().clone() for k, v in self.adapter.state_dict().items()}
        self.optimizer.zero_grad(set_to_none=True)
        new_norm = sample_chunk(self.extractor, self.policy, observation_after, z0,
                                context, return_normalized=True)
        new_norm_before = new_norm.detach().clone()
        old_suffix = self.reference_normalized[:, executed_k:, :]
        new_prefix = new_norm[:, :h - executed_k, :]
        consistency = torch.nn.functional.mse_loss(new_prefix, old_suffix.detach())
        param_anchor = sum((p - self.initial_params[k]).pow(2).mean()
                           for k, p in self.adapter.named_parameters())
        loss = consistency + self.parameter_anchor_weight * param_anchor
        if not torch.isfinite(loss):
            return {"updated": False, "reason": "non_finite_loss", "loss_before": loss_before}
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.max_grad_norm)
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            return {"updated": False, "reason": "non_finite_grad", "loss_before": loss_before}
        self.optimizer.step()
        all_finite = all(bool(torch.isfinite(p).all()) for p in self.adapter.parameters())
        changed = any(not torch.equal(p.detach(), before[k])
                      for k, p in self.adapter.named_parameters())
        if not all_finite:
            self.adapter.load_state_dict(before)
            self.optimizer = torch.optim.Adam(self.adapter.parameters(),
                                              lr=self.optimizer.param_groups[0]["lr"])
            return {"updated": False, "reason": "non_finite_param", "loss_before": loss_before}
        loss_after = float(self._loss_only(observation_after, z0, context, executed_k))
        with torch.no_grad():
            post_norm = sample_chunk(self.extractor, self.policy, observation_after, z0,
                                     context, return_normalized=True)
        action_delta = float((post_norm - new_norm_before).abs().mean())
        sq = 0.0; numel = 0
        for k, p in self.adapter.named_parameters():
            d = p.detach() - before[k]
            sq += float(d.pow(2).sum()); numel += d.numel()
        param_l2 = sq ** 0.5
        param_rms = (sq / max(1, numel)) ** 0.5
        return {"updated": bool(changed), "loss_before": loss_before,
                "loss_after": loss_after, "delta_loss": loss_after - loss_before,
                "consistency_before_update": float(consistency.detach()),
                "action_delta": action_delta, "grad_norm": float(grad_norm),
                "param_delta_l2": param_l2, "param_delta_rms": param_rms}
