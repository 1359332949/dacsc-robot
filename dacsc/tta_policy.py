"""Inference and update paths for the first action-conditioned visual TTA."""

from __future__ import annotations
import types
import torch
from corp.preprocessing import prepare_observation
from .visual_adapter import AdaptedRepresentationExtractor


class TTAPolicy:
    """Frozen CoRP with a trainable residual adapter before the Resampler."""

    def __init__(self, policy, adapter) -> None:
        self.policy = policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        device = next(policy.parameters()).device
        self.adapter = adapter.to(device=device)
        self.extractor = AdaptedRepresentationExtractor(policy.representation_extractor, self.adapter)

    @torch.no_grad()
    def predict(self, observation, num_steps=None, context=None):
        batch = prepare_observation(
            observation, config=self.policy.config, normalizer=self.policy.normalizer,
            device=next(self.policy.parameters()).device)
        representation = self.extractor(
            batch["images"], batch["state"], batch["instruction"], context=context)
        actions = self.policy.action_generator.sample(representation, num_steps=num_steps)
        return self.policy.normalizer.unnormalize_action(actions)

    def encode_visual_for_update(self, observation, context=None):
        """Differentiable visual encoding used only after an executed chunk."""
        batch = prepare_observation(
            observation, config=self.policy.config, normalizer=self.policy.normalizer,
            device=next(self.policy.parameters()).device)
        ext = self.policy.representation_extractor
        device = next(self.policy.parameters()).device
        views = tuple(ext.view_names)
        images = batch["images"]
        patches = ext.visual_encoder(torch.cat([images[v] for v in views], dim=0))
        batch_size = images[views[0]].shape[0]
        patch_tokens = {v: patches[i * batch_size:(i + 1) * batch_size]
                        for i, v in enumerate(views)}
        adapted = {v: self.adapter(t, v, context) for v, t in patch_tokens.items()}
        language_tokens, pooled = ext.language_encoder(batch["instruction"])
        visual = ext.cross_attention_resampler(adapted, pooled)
        state_token = ext.state_encoder(batch["state"])
        representation = types.SimpleNamespace(
            tokens=torch.cat([visual, language_tokens.to(visual.dtype), state_token], dim=1),
            pooled_language=pooled, action_dtype=visual.dtype)
        return {"z": visual.mean(dim=1), "visual": visual,
                "representation": representation, "patch_tokens": patch_tokens,
                "adapted_patch_tokens": adapted}
