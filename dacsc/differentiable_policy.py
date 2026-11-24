"""Differentiable inference path through a frozen CoRP policy.

The visual encoder, resampler, language/state encoders and action generator
weights stay frozen. Only an external adapter receives gradients.
"""

from __future__ import annotations
from dataclasses import replace
import torch

from corp.representation.extractor import Representation


def _encode_frozen(policy, observation):
    """Run the frozen representation stack without grad; return its pieces."""
    from corp.preprocessing import prepare_observation
    with torch.no_grad():
        batch = prepare_observation(
            observation,
            config=policy.config,
            normalizer=policy.normalizer,
            device=next(policy.parameters()).device,
        )
        extractor = policy.representation_extractor
        patches = extractor.visual_encoder(
            torch.cat([batch["images"][n] for n in extractor.view_names], dim=0)
        )
        bs = batch["state"].shape[0]
        patch_tokens = {
            name: patches[i * bs:(i + 1) * bs]
            for i, name in enumerate(extractor.view_names)
        }
        language_tokens, pooled_language = extractor.language_encoder(batch["instruction"])
        visual_tokens = extractor.cross_attention_resampler(patch_tokens, pooled_language)
        state_token = extractor.state_encoder(batch["state"])
    return {
        "visual_tokens": visual_tokens,
        "language_tokens": language_tokens,
        "state_token": state_token,
        "pooled_language": pooled_language,
    }


def sample_from_z0(policy, representation: Representation, z0: torch.Tensor,
                   num_steps: int | None = None) -> torch.Tensor:
    """Differentiable Euler flow-matching sampler with an explicit z0."""
    gen = policy.action_generator
    steps = gen.num_sample_steps if num_steps is None else int(num_steps)
    if steps <= 0:
        raise ValueError("num_steps must be positive")
    if z0.shape != (representation.tokens.shape[0], gen.action_horizon, gen.action_dim):
        raise ValueError(f"z0 shape {tuple(z0.shape)} does not match action shape")
    if z0.device != representation.tokens.device:
        z0 = z0.to(representation.tokens.device)
    action = z0.clone()
    prefix = gen.prepare_prefix(representation)
    for step in range(steps):
        tau = torch.full((action.shape[0],), step / float(steps),
                         device=action.device, dtype=action.dtype)
        velocity = gen(action, tau, prefix)
        action = action + velocity / float(steps)
    return action


def forward_with_adapter(policy, observation, adapter, z0: torch.Tensor,
                         num_steps: int | None = None):
    """Return normalized and controller-space actions plus frozen tokens."""
    pieces = _encode_frozen(policy, observation)
    adapted_visual = adapter(pieces["visual_tokens"])
    visual_language = torch.cat(
        [adapted_visual, pieces["language_tokens"].to(adapted_visual.dtype)], dim=1
    )
    representation = Representation(
        tokens=torch.cat([visual_language, pieces["state_token"]], dim=1),
        pooled_language=pieces["pooled_language"],
        action_dtype=visual_language.dtype,
    )
    normalized = sample_from_z0(policy, representation, z0, num_steps=num_steps)
    controller = policy.normalizer.unnormalize_action(normalized)
    return {"normalized_actions": normalized, "controller_actions": controller,
            "visual_tokens": pieces["visual_tokens"], "adapted_tokens": adapted_visual}


def zero_init_equivalence(policy, observation, adapter, z0: torch.Tensor,
                          num_steps: int | None = None) -> dict:
    """Check adapter=identity and frozen forward paths agree."""
    with torch.no_grad():
        pieces = _encode_frozen(policy, observation)
        visual_language = torch.cat(
            [pieces["visual_tokens"], pieces["language_tokens"].to(pieces["visual_tokens"].dtype)], dim=1
        )
        rep = Representation(
            tokens=torch.cat([visual_language, pieces["state_token"]], dim=1),
            pooled_language=pieces["pooled_language"],
            action_dtype=visual_language.dtype,
        )
        base = sample_from_z0(policy, rep, z0, num_steps=num_steps)
        out = forward_with_adapter(policy, observation, adapter, z0, num_steps=num_steps)["normalized_actions"]
    return {"max_abs_diff": float((base - out).abs().max().item()),
            "shapes": [tuple(base.shape), tuple(out.shape)]}
