"""Source-calibrated dual-view gate; repeat vision only, preserve native decoder.

The first native embed_prefix call is intercepted after native preprocessing and
Flow-noise creation. Both unadapted vision outputs are computed, then either
reused or replaced by a second shallow-adapted vision pass. The original prefix
builder subsequently runs its connectors/language/state embedding exactly once.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from dacsc.smolvla_v1_hook import SmolVLAV1Hook


class _ScoreHook(SmolVLAV1Hook):
    def _hook(self, layer):
        parent = super()._hook(layer)

        def capture(module, args, output):
            view = self.counts[layer]
            x = output[0] if isinstance(output, tuple) else output
            result = parent(module, args, output)
            if layer == 12:
                mu, ss, tau = (v[view].float()[None] for v in self.references[12])
                p = x.float()
                d2 = ((p.mean(1) - mu).square() + (p.std(1, unbiased=False) - ss).square()) / (ss.square() + 1e-6)
                r = (d2 / tau.clamp_min(1e-6)).mean()
                if not torch.isfinite(r):
                    raise ValueError('nonfinite deep gate score')
                self.stats[layer, view]['normalized_mean'] = r.detach()
            return result

        return capture


class SmolVLADeepGate:
    """One original actor.act per replan, with no extra RNG draw or queue update.

    decision_override is only a preflight oracle; actual evaluations use None.
    audit_inputs enables full cached-pixel checks in preflight, not a new method.
    """
    implementation_id = 'smolvla_source_deep12_shallow3_v1'

    def __init__(self, policy, references, threshold, *, audit_inputs=False, decision_override=None):
        self.policy = policy
        self.core = policy.model
        self.vision = self.core.vlm_with_expert.vlm.model.vision_model
        self.references = references
        self.threshold = float(threshold)
        self.audit_inputs = audit_inputs
        self.decision_override = decision_override
        self.original_prefix = self.core.embed_prefix
        self.patch = patch.object(self.core, 'embed_prefix', new=self._prefix)
        self.patch.start()
        self.open = False
        self.last = None
        self.calls = 0

    def begin(self):
        if self.open:
            raise RuntimeError('unfinished deep-gate replan')
        self.open = True
        self.last = None
        self.calls = 0

    def _vision_pass(self, images, active):
        h = _ScoreHook(self.policy, self.references, layers=(3, 12), active_layer=3 if active else None, alpha=.75)
        try:
            h.begin()
            outputs = [self.vision(pixel_values=im.to(dtype=self.vision.dtype), patch_attention_mask=None) for im in images]
            stats = h.end()
            assert len(outputs) == 2 and all(o.last_hidden_state.dtype == torch.bfloat16 for o in outputs)
            return outputs, stats
        finally:
            h.close()

    def _prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        if not self.open or self.calls:
            raise RuntimeError('deep gate requires exactly one native prefix per replan')
        self.calls += 1
        if len(images) != 2 or len(img_masks) != 2:
            raise ValueError('expected two native serial camera streams')
        selected, raw = self._vision_pass(images, active=False)
        score = float(np.mean([float(raw[12, v]['normalized_mean']) for v in range(2)]))
        accepted = score > self.threshold if self.decision_override is None else self.decision_override
        assert type(accepted) is bool
        if accepted:
            # Release the unadapted tokens before allocating the second pass.
            selected = None
            selected, _ = self._vision_pass(images, active=True)
        self.last = dict(score=score, threshold=self.threshold, accepted=accepted,
            raw_active_layer3=[float(raw[3, v]['active_fraction']) for v in range(2)],
            raw_active_layer12=[float(raw[12, v]['active_fraction']) for v in range(2)],
            raw_layer12_R=[float(raw[12, v]['normalized_mean']) for v in range(2)],
            encoder_calls=4 if accepted else 2, prefix_builds=1,
            decision_rule='source' if self.decision_override is None else 'preflight_override')
        cursor = 0

        def cached_forward(pixel_values, patch_attention_mask=None, **kwargs):
            nonlocal cursor
            if cursor >= 2 or patch_attention_mask is not None or kwargs:
                raise RuntimeError('native vision-call interface changed')
            expected = images[cursor]
            assert pixel_values.shape == expected.shape and pixel_values.dtype == self.vision.dtype
            assert pixel_values.device == expected.device
            if self.audit_inputs:
                assert torch.equal(pixel_values, expected.to(dtype=self.vision.dtype)), 'cached camera pixels/order mismatch'
            result = selected[cursor]
            cursor += 1
            return result

        # Only vision.forward is cached. Native connector, language model and
        # all ten Flow steps remain on their original unmodified code path.
        with patch.object(self.vision, 'forward', new=cached_forward):
            prefix = self.original_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        assert cursor == 2
        return prefix

    def end(self):
        if not self.open or self.calls != 1 or self.last is None:
            raise RuntimeError('missing native deep-gate replan')
        self.open = False
        return dict(self.last)

    def close(self):
        self.patch.stop()
        self.open = False
