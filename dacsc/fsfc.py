"""F-SFC: source-only channel priors and episode-local gate proposals (no simulator)."""
from dataclasses import dataclass
import math

import torch

VIEWS = ('agentview', 'eye_in_hand')
ARMS = ('jacobian', 'base_std', 'uniform', 'jacobian_hard_mask')


def source_ids(cases, task):
    # Filter by provenance before image loading or computing any statistics.
    rows = [c for c in cases if c['role'] == 'dev_clean' and c['factor'] == 'clean'
            and c['base_task'] != task]
    if (len(rows) != 9 or len({c['base_task'] for c in rows}) != 9
            or len({c['case_id'] for c in rows}) != 9):
        raise ValueError('expected one Clean reference for each of nine other tasks')
    return sorted(c['case_id'] for c in rows)


def frobenius(action_fn, patches, noises):
    """Exact E_z ||d a / d P[:,c]||_F. No cancellation between action coordinates."""
    p = patches.detach().requires_grad_(True)  # [view,patch,channel]
    norms = []
    for z in noises.split(1):
        actions = action_fn(p, z).reshape(-1)
        sq = torch.zeros_like(p[:, 0])
        for j, action in enumerate(actions.unbind()):
            grad, = torch.autograd.grad(action, p, retain_graph=j+1 < len(actions))
            sq += grad.detach().square().sum(1)
        norms.append(sq.sqrt())
    result = torch.stack(norms).mean(0)
    if not torch.isfinite(result).all():
        raise ValueError('nonfinite source Jacobian')
    return result


@dataclass(frozen=True)
class Prior:
    mean: torch.Tensor
    mean_scale: torch.Tensor
    rho: torch.Tensor
    patch_scale: torch.Tensor | None = None


def fit_priors(clean, drift, importance, eps=1e-6):
    """Inputs [scene,view,patch,channel], [scene,view,channel] x 2; equal scene weights."""
    if (clean.ndim != 4 or clean.shape[1] != 2 or drift.shape != importance.shape
            or drift.shape != clean.shape[:2]+clean.shape[-1:]):
        raise ValueError('invalid prior shapes')
    if not all(torch.isfinite(t).all() for t in (clean, drift, importance)) or (drift<0).any() or (importance<0).any():
        raise ValueError('invalid source evidence')
    mu = clean.mean((0, 2)); patch_scale = clean.std((0, 2), unbiased=False) + eps
    var = clean.mean(2).var(0, unbiased=False)
    mean_scale = (.9*var + .1*var.mean()).sqrt() + eps
    nuisance = drift.mean(0) / patch_scale
    old = clean.std(0, unbiased=False).mean(1)
    weights = {'jacobian': nuisance/(importance.mean(0)+eps),
               'base_std': nuisance/(old+eps), 'uniform': torch.ones_like(mu)}
    return {k: Prior(mu.detach(), mean_scale.detach(), (v/v.mean() if v.mean()>0 else v).detach(), patch_scale.detach())
            for k,v in weights.items()}


@dataclass(frozen=True)
class Config:
    hard_mask: bool = False
    tau: float = 2.
    lam: float = 1.
    eta: float = .01
    delta: float = .001
    lr: float = .1


def eligible(p, prior, cfg):
    mask = prior.rho > 0
    if cfg.hard_mask:
        mask = mask & ((p.mean(1)-prior.mean).abs()/prior.mean_scale > cfg.tau)
    return mask


def transform(p, g, prior):
    return p + g[:, None, :] * (prior.mean[:, None, :] - p)


def loss(p, g, prior, z, curvature, cfg):
    q = transform(p, g, prior)
    c = curvature(q, z)
    d = (((q.mean(1)-prior.mean)/prior.mean_scale).abs()-cfg.tau).relu()
    l = c + cfg.lam*(prior.rho*d.square()).mean() + cfg.eta*(g.square()/(prior.rho+cfg.delta)).mean()
    return l,c


def accepts(l0,c0,l1,c1):
    return all(math.isfinite(x) and x>=0 for x in (l0,c0,l1,c1)) and l0>0 and l1<=.95*l0 and c1<=1.05*c0


def propose(p, prior, train_z, validation_z, curvature, cfg=Config(), *, previous=None, steps=10):
    """No labels, Clean twin or reference actions. A rejected update applies exactly zero."""
    if train_z.shape != validation_z.shape or train_z.shape[0] != 4:
        raise ValueError('four fit and four validation noises required')
    noises = torch.cat((train_z,validation_z)).flatten(1)
    if len(noises.unique(dim=0)) != 8 or not noises.isfinite().all():
        raise ValueError('reused or invalid noise')
    p = p.detach(); mask = eligible(p, prior, cfg); zero = torch.zeros_like(prior.mean)
    if not mask.any():
        return zero, {'accepted':False,'reason':'dead_zone','active_channels':0}
    g = (zero.clone() if previous is None else previous.detach().clone()*mask).requires_grad_()
    opt = torch.optim.Adam([g],lr=cfg.lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        total,_ = loss(p,g*mask,prior,train_z,curvature,cfg)
        grad, = torch.autograd.grad(total,g)
        if not torch.isfinite(total) or not grad.isfinite().all():
            return zero, {'accepted':False,'reason':'nonfinite_fit'}
        g.grad=grad; opt.step()
        with torch.no_grad():
            g.clamp_(0.,1.).mul_(mask)
    with torch.no_grad():
        b,c=loss(p,zero,prior,validation_z,curvature,cfg)
        u,v=loss(p,g,prior,validation_z,curvature,cfg)
        values=list(map(float,(b,c,u,v))); ok=accepts(*values)
    return (g.detach() if ok else zero), {'accepted':ok,'reason':'accepted' if ok else 'rollback',
        'active_channels':int(mask.sum()),'candidate_max':float(g.detach().max()),
        'validation_L0_C0_L1_C1':[x if math.isfinite(x) else None for x in values]}


class EpisodeGate:
    """Optional future schedule, not enabled in t=0 runner. Create a new instance per episode."""
    def __init__(self):
        self.g=None; self.last=-1

    def step(self, chunk, p, prior, train_z, validation_z, curvature, cfg=Config()):
        if chunk != self.last+1:
            raise ValueError('expected consecutive chunks beginning at zero')
        self.last=chunk
        if chunk==0:
            self.g,info=propose(p,prior,train_z,validation_z,curvature,cfg)
        else:
            self.g=self.g*eligible(p,prior,cfg)  # apply the selected arm’s eligibility rule
            info={'reason':'reuse'}
            if chunk%2==0:
                with torch.no_grad():
                    prev,_=loss(p,self.g,prior,validation_z,curvature,cfg)
                    base,_=loss(p,torch.zeros_like(self.g),prior,validation_z,curvature,cfg)
                # Refresh when the old gate no longer improves the current identity objective.
                if not torch.isfinite(prev) or not torch.isfinite(base) or prev>=base:
                    self.g,info=propose(p,prior,train_z,validation_z,curvature,cfg,previous=self.g,steps=3)
        return self.g.clone(),info
