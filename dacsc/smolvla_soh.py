"""SOH-VLA: deep-gated orthogonal-subspace hypernetwork for frozen SmolVLA.

Design contract (matches the audited ``SmolVLAV1Hook`` / ``SmolVLADeepGate``):

* the native SmolVLA weights are frozen and are never updated here;
* pass 1 is the unadapted dual-camera SigLIP forward;
* the Layer-12 source-scaled statistic is the same gate used by
  ``smolvla_deepgate.py``; below threshold, pass-1 tokens are reused exactly;
* above threshold, only Layer-3 tokens are repaired, and only in the
  orthogonal complement of the source action-sensitivity basis;
* the native connector / language model / Flow decoder still run once.

This module intentionally does *not* claim that orthogonal projection is a
physical guarantee.  ``U_geo`` is an estimate from source-domain gradients;
the projection preserves the estimated action-sensitive subspace only.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dacsc.smolvla_v1_hook import SmolVLAV1Hook


def spatial_hf_ratio(p: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    """Scale-invariant 32x32 patch-grid high-frequency sharpness.

    ``p`` is [B, 1024, 768] in FP32.  The 1024 SigLIP patches are arranged as
    a 32x32 grid.  A blur/fog image loses high-frequency energy, so the ratio
    falls relative to the Clean source standard deviation.
    """
    b, n, c = p.shape
    if n != 1024:
        raise ValueError(f'expected 1024 SigLIP patches, got {n}')
    grid = p.view(b, 32, 32, c)
    dx = (grid[:, :, 1:, :] - grid[:, :, :-1, :]).square().mean(dim=(1, 2))
    dy = (grid[:, 1:, :, :] - grid[:, :-1, :, :]).square().mean(dim=(1, 2))
    return (dx + dy + 1e-8).sqrt() / (sd + 1e-6)


class SOHModule(nn.Module):
    """Basis-mixed low-rank residual operator in the domain-sensitive space.

    Parameters: ~1.2M for dim=768, rank=16, M=8, z_dim=128.  The residual
    scale is initialized to zero, so a fresh module is exactly the frozen
    network (bit-identical when ``scale == 0``).
    """

    def __init__(
        self,
        u_geo: torch.Tensor,
        dim: int = 768,
        rank: int = 16,
        num_bases: int = 8,
        z_dim: int = 128,
    ):
        super().__init__()
        if u_geo.ndim != 2 or u_geo.shape[0] != dim:
            raise ValueError(f'u_geo must be [dim, k], got {tuple(u_geo.shape)}')
        if u_geo.shape[1] == 0 or u_geo.shape[1] > dim:
            raise ValueError('u_geo has invalid number of columns')
        u_geo = u_geo.float()
        if not torch.allclose(u_geo.T @ u_geo, torch.eye(u_geo.shape[1], device=u_geo.device, dtype=u_geo.dtype), atol=1e-4):
            raise ValueError('u_geo columns must be orthonormal')
        self.register_buffer('u_geo', u_geo.contiguous())
        self.dim = int(dim)
        self.rank = int(rank)
        self.num_bases = int(num_bases)

        # [d_mu3, d_sd3, hf3, d_mu12, d_sd12] = 5 * 768 = 3840
        self.encoder = nn.Sequential(
            nn.Linear(dim * 5, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, z_dim),
            nn.LayerNorm(z_dim),
            nn.GELU(),
        )
        self.head_w = nn.Linear(z_dim, num_bases)
        self.head_gamma = nn.Linear(z_dim, rank)
        self.a_basis = nn.Parameter(torch.randn(num_bases, dim, rank) * 0.02)
        self.b_basis = nn.Parameter(torch.randn(num_bases, rank, dim) * 0.02)
        self.scale = nn.Parameter(torch.zeros(1))

    def project_dom(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x P_dom`` with ``P_dom = I - U_geo U_geo^T``."""
        # x: [B, N, D]; u_geo: [D, k]; no D x D matrix is materialized.
        return x - (x @ self.u_geo) @ self.u_geo.T

    @staticmethod
    def build_raw_fingerprint(
        p3: torch.Tensor,
        p12: torch.Tensor,
        ref3: tuple[torch.Tensor, torch.Tensor],
        ref12: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Build [B, 5*D] domain fingerprint in FP32.

        ``p3`` / ``p12`` are [B, 1024, D]; ``ref3`` / ``ref12`` are per-view
        source ``(mean, std)`` vectors of shape [D].
        """
        if p3.ndim != 3 or p12.ndim != 3:
            raise ValueError('patch features must be [B, N, D]')
        if p3.shape[0] != p12.shape[0] or p3.shape[-1] != p12.shape[-1]:
            raise ValueError('Layer-3 and Layer-12 batch/channel mismatch')
        if ref3[0].shape != ref3[1].shape or ref12[0].shape != ref12[1].shape:
            raise ValueError('reference mean/std mismatch')
        mu3 = ref3[0].float()[None]
        ss3 = ref3[1].float()[None]
        mu12 = ref12[0].float()[None]
        ss12 = ref12[1].float()[None]

        m3 = p3.float().mean(1)
        sd3 = p3.float().std(1, unbiased=False)
        m12 = p12.float().mean(1)
        sd12 = p12.float().std(1, unbiased=False)

        d_mu3 = ((m3 - mu3) / (ss3 + 1e-6)).clamp(-10.0, 10.0)
        d_sd3 = torch.log((sd3 + 1e-6) / (ss3 + 1e-6)).clamp(-5.0, 5.0)
        hf3 = spatial_hf_ratio(p3.float(), sd3).clamp(0.0, 10.0)
        d_mu12 = ((m12 - mu12) / (ss12 + 1e-6)).clamp(-10.0, 10.0)
        d_sd12 = torch.log((sd12 + 1e-6) / (ss12 + 1e-6)).clamp(-5.0, 5.0)
        return torch.cat([d_mu3, d_sd3, hf3, d_mu12, d_sd12], dim=-1)

    def forward(self, x3: torch.Tensor, raw_fp: torch.Tensor) -> torch.Tensor:
        """Repair Layer-3 tokens; return original dtype."""
        if x3.ndim != 3 or x3.shape[-1] != self.dim:
            raise ValueError(f'x3 must be [B, N, {self.dim}], got {tuple(x3.shape)}')
        if raw_fp.ndim != 2 or raw_fp.shape[0] != x3.shape[0] or raw_fp.shape[1] != self.dim * 5:
            raise ValueError(f'raw_fp must be [B, {self.dim * 5}], got {tuple(raw_fp.shape)}')
        orig_dtype = x3.dtype
        p = x3.float()
        z = self.encoder(raw_fp.float())
        w = F.softmax(self.head_w(z), dim=-1)                    # [B, M]
        gamma = 1.0 + torch.tanh(self.head_gamma(z))             # [B, r]

        a_t = torch.einsum('bm,mdr->bdr', w, self.a_basis)       # [B, D, r]
        b_t = torch.einsum('bm,mrd->brd', w, self.b_basis)       # [B, r, D]
        x_dom = self.project_dom(p)
        h = F.gelu(torch.bmm(x_dom, a_t)) * gamma[:, None, :]
        delta = self.project_dom(torch.bmm(h, b_t))
        return (p + self.scale * delta).to(orig_dtype)


class _SOHProbeHook(SmolVLAV1Hook):
    """Unadapted pass: capture Layer-3/12 token statistics and raw features."""

    def _hook(self, layer):
        parent = super()._hook(layer)

        def capture(module, args, output):
            view = self.counts[layer]
            x = output[0] if isinstance(output, tuple) else output
            result = parent(module, args, output)
            p = x.float()
            self.stats[layer, view]['feat'] = p.detach()
            if layer == 12:
                mu, ss, tau = (v[view].float()[None] for v in self.references[12])
                d2 = ((p.mean(1) - mu).square() + (p.std(1, unbiased=False) - ss).square()) / (ss.square() + 1e-6)
                r = (d2 / tau.clamp_min(1e-6)).mean()
                if not torch.isfinite(r):
                    raise ValueError('nonfinite SOH gate score')
                self.stats[layer, view]['normalized_mean'] = r.detach()
            return result

        return capture


class SmolVLASOHAdapter:
    """Native SmolVLA adapter: deep gate + streaming SOH repair.

    Contract matches ``SmolVLADeepGate``: call ``begin()`` immediately before
    ``actor.act`` and ``end()`` immediately after.  ``reset_episode()`` clears
    the per-camera streaming EMA at episode start.
    """

    implementation_id = 'smolvla_soh_streaming_v1'

    def __init__(
        self,
        policy,
        references,
        threshold: float,
        soh_module: SOHModule,
        *,
        ema_beta: float = 0.8,
        audit_inputs: bool = False,
        decision_override=None,
    ):
        if not 0.0 <= float(ema_beta) < 1.0:
            raise ValueError('ema_beta must be in [0, 1)')
        self.policy = policy
        self.core = policy.model
        self.vision = self.core.vlm_with_expert.vlm.model.vision_model
        self.references = references
        self.threshold = float(threshold)
        self.soh = soh_module.eval()
        self.ema_beta = float(ema_beta)
        self.audit_inputs = bool(audit_inputs)
        self.decision_override = decision_override

        self.original_prefix = self.core.embed_prefix
        self.patch = patch.object(self.core, 'embed_prefix', new=self._prefix)
        self.patch.start()

        self.open = False
        self.last = None
        self.calls = 0
        self.ema_fp = [None, None]

    def reset_episode(self):
        self.ema_fp = [None, None]

    def update_ema(self, view: int, fp_now: torch.Tensor) -> torch.Tensor:
        """Streaming per-camera EMA with a Step-0 first-frame initializer."""
        if view not in (0, 1):
            raise ValueError('view must be 0 or 1')
        if self.ema_fp[view] is None:
            self.ema_fp[view] = fp_now
        else:
            self.ema_fp[view] = self.ema_beta * self.ema_fp[view] + (1.0 - self.ema_beta) * fp_now
        return self.ema_fp[view]

    def begin(self):
        if self.open:
            raise RuntimeError('unfinished SOH replan')
        self.open = True
        self.last = None
        self.calls = 0

    def _pass1_probe(self, images):
        h = _SOHProbeHook(self.policy, self.references, layers=(3, 12), active_layer=None)
        try:
            h.begin()
            outputs = [
                self.vision(pixel_values=im.to(dtype=self.vision.dtype), patch_attention_mask=None)
                for im in images
            ]
            stats = h.end()
            if len(outputs) != 2:
                raise RuntimeError('expected two native camera calls')
            return outputs, stats
        finally:
            h.close()

    def _pass2_soh(self, images, fps):
        view_idx = 0

        def l3_hook(module, args, output):
            nonlocal view_idx
            x = output[0] if isinstance(output, tuple) else output
            fixed = self.soh(x, fps[view_idx])
            view_idx += 1
            return (fixed, *output[1:]) if isinstance(output, tuple) else fixed

        handle = self.vision.encoder.layers[2].register_forward_hook(l3_hook)
        try:
            with torch.no_grad():
                outputs = [
                    self.vision(pixel_values=im.to(dtype=self.vision.dtype), patch_attention_mask=None)
                    for im in images
                ]
            if view_idx != 2:
                raise RuntimeError('expected two SOH repair calls')
            return outputs
        finally:
            handle.remove()

    def _prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        if not self.open or self.calls:
            raise RuntimeError('SOH requires exactly one native prefix per replan')
        if len(images) != 2 or len(img_masks) != 2:
            raise ValueError('expected two native serial camera streams')
        self.calls += 1

        selected, raw = self._pass1_probe(images)
        score = float(np.mean([float(raw[12, v]['normalized_mean']) for v in range(2)]))
        accepted = score > self.threshold if self.decision_override is None else self.decision_override
        if type(accepted) is not bool:
            raise TypeError('SOH gate decision must be bool')

        if accepted:
            fps = []
            for v in range(2):
                fp_now = SOHModule.build_raw_fingerprint(
                    raw[3, v]['feat'],
                    raw[12, v]['feat'],
                    (self.references[3][0][v], self.references[3][1][v]),
                    (self.references[12][0][v], self.references[12][1][v]),
                )
                fps.append(self.update_ema(v, fp_now))
            selected = None
            selected = self._pass2_soh(images, fps)

        self.last = dict(
            score=score,
            threshold=self.threshold,
            accepted=bool(accepted),
            encoder_calls=4 if accepted else 2,
            prefix_builds=1,
            decision_rule='source' if self.decision_override is None else 'preflight_override',
            ema_beta=self.ema_beta,
        )

        cursor = 0

        def cached_forward(pixel_values, patch_attention_mask=None, **kwargs):
            nonlocal cursor
            if cursor >= 2 or patch_attention_mask is not None or kwargs:
                raise RuntimeError('native vision-call interface changed')
            expected = images[cursor]
            if pixel_values.shape != expected.shape or pixel_values.dtype != self.vision.dtype:
                raise ValueError('cached camera pixel/dtype mismatch')
            if pixel_values.device != expected.device:
                raise ValueError('cached camera device mismatch')
            if self.audit_inputs and not torch.equal(pixel_values, expected.to(dtype=self.vision.dtype)):
                raise AssertionError('cached camera pixels/order mismatch')
            result = selected[cursor]
            cursor += 1
            return result

        with patch.object(self.vision, 'forward', new=cached_forward):
            prefix = self.original_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        if cursor != 2:
            raise RuntimeError('native prefix did not consume both camera streams')
        return prefix

    def end(self):
        if not self.open or self.calls != 1 or self.last is None:
            raise RuntimeError('missing native SOH replan')
        self.open = False
        return dict(self.last)

    def close(self):
        self.patch.stop()
        self.open = False


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
