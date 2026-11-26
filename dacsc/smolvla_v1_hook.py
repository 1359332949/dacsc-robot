"""Single-layer V1 for native BF16 SmolVLA. No CLS, two serial camera calls.

Statistics are FP32. This is the existing bounded source-scaled moment operator,
not an exact unconstrained Wasserstein transport or a guarantee of geometry.
"""
from __future__ import annotations
import torch

LAYERS = (3, 6, 9, 12)
VIEWS = ('agentview', 'eye_in_hand')


def moments(x):
    p = x.float()
    return p.mean(1), p.square().mean(1)


def align_tokens(x, reference, alpha=0.75):
    if alpha == 0:
        return x
    if not 0 < alpha <= 1:
        raise ValueError('alpha outside [0,1]')
    mu, ss, tau = (v.float() for v in reference)
    p = x.float()
    m = p.mean(1)
    sd = p.std(1, unbiased=False)
    d2 = ((m - mu).square() + (sd - ss).square()) / (ss.square() + 1e-6)
    active = d2 > tau
    gain = (((1 - alpha) * sd + alpha * ss) / (sd.square() + 1e-6).sqrt()).clamp(.5, 2.)
    corrected = gain[:, None, :] * (p - m[:, None, :]) + ((1-alpha)*m + alpha*mu)[:, None, :]
    return torch.where(active[:, None, :], corrected, p).to(x.dtype)


def fit_loto(raw):
    """raw[task]={'mean','second'}: [frames,4,2,channels]. No outcome inputs.

    Pool actual source frames equally, matching the frozen CoRP full-trajectory
    reference. Outer target task excluded from every inner threshold fold.
    """
    tasks = sorted(raw)
    if len(tasks) != 10:
        raise ValueError('exactly ten source tasks required')
    refs, audit = {}, {}
    for target in tasks:
        outer = [t for t in tasks if t != target]
        assert len(outer) == 9
        audit[target] = {'outer': outer, 'inner': {h: [t for t in outer if t != h] for h in outer}}
        for li, layer in enumerate(LAYERS):
            def pooled(names):
                m = torch.cat([raw[t]['mean'][:, li].float() for t in names])
                q = torch.cat([raw[t]['second'][:, li].float() for t in names])
                mu = m.mean(0)
                return mu, (q.mean(0)-mu.square()).clamp_min(0).sqrt()
            mu, ss = pooled(outer)
            scores = []
            for hold in outer:
                inner = audit[target]['inner'][hold]
                assert len(inner) == 8 and target not in inner and hold not in inner
                mi, si = pooled(inner)
                m, q = raw[hold]['mean'][:, li].float(), raw[hold]['second'][:, li].float()
                sd = (q - m.square()).clamp_min(0).sqrt()
                scores.append(((m-mi).square()+(sd-si).square())/(si.square()+1e-6))
            tau = torch.cat(scores).quantile(.95, dim=0)
            refs[target, layer] = (mu, ss, tau)
            assert all(torch.isfinite(v).all() for v in (mu, ss, tau))
    return refs, audit


class SmolVLAV1Hook:
    """begin()/end() delimit one native replan; each layer must see A then W.

    Call order is checked against native prepare_images order in the real-model
    preflight. Independent counters per layer avoid mixing layer/view dimensions.
    """
    def __init__(self, policy, references=None, layers=LAYERS, active_layer=None, alpha=.75):
        self.policy = policy
        self.vision = policy.model.vlm_with_expert.vlm.model.vision_model
        assert len(self.vision.encoder.layers) == 12
        assert list(policy.config.image_features) == ['observation.images.image', 'observation.images.image2']
        self.layers = tuple(layers)
        self.references = references or {}
        self.active_layer, self.alpha = active_layer, alpha
        self.handles = [self.vision.encoder.layers[l-1].register_forward_hook(self._hook(l)) for l in self.layers]
        self.open = False
        self.counts = {}
        self.stats = {}

    def begin(self):
        if self.open:
            raise RuntimeError('nested or unfinished replan')
        self.open = True
        self.counts = {l: 0 for l in self.layers}
        self.stats = {}

    def _hook(self, layer):
        def capture(module, args, output):
            if not self.open:
                raise RuntimeError('vision call outside delimited replan')
            view = self.counts[layer]
            if view >= 2:
                raise RuntimeError('more than two camera calls')
            self.counts[layer] += 1
            x = output[0] if isinstance(output, tuple) else output
            if not isinstance(x, torch.Tensor) or tuple(x.shape) != (1, 1024, 768):
                raise ValueError('native SigLIP patch/batch interface changed')
            if x.dtype != torch.bfloat16:
                raise ValueError('native vision BF16 contract changed')
            m, q = moments(x)
            entry = {'mean': m[0].detach(), 'second': q[0].detach()}
            if layer in self.references:
                mu, ss, tau = (v[view].float()[None] for v in self.references[layer])
                sd = x.float().std(1, unbiased=False)
                d2 = ((m-mu).square()+(sd-ss).square())/(ss.square()+1e-6)
                active = d2 > tau
                gain = (((1-self.alpha)*sd+self.alpha*ss)/(sd.square()+1e-6).sqrt()).clamp(.5,2.)
                entry['active_fraction'] = active.float().mean().detach()
                entry['gain_abs_mean'] = torch.where(active, (gain-1).abs(), 0.).mean().detach()
                if layer == self.active_layer:
                    changed = align_tokens(x, (mu, ss, tau), self.alpha)
                    assert changed.dtype == x.dtype
                    self.stats[layer, view] = entry
                    if isinstance(output, tuple):
                        return (changed, *output[1:])
                    return changed
            self.stats[layer, view] = entry
            return None
        return capture

    def end(self):
        if not self.open or any(self.counts.get(l) != 2 for l in self.layers):
            raise RuntimeError(f'each replan needs exactly two calls per layer: {self.counts}')
        self.open = False
        return self.stats

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.open = False
