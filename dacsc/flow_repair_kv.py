"""Post-LayerNorm cache-Value view-trust flow intervention primitives. No Clean/reference inputs."""
from types import SimpleNamespace
from dataclasses import replace
import torch
from dacsc.spatial_patch_bias import resampler_forward_with_bias


def integrate(policy,representation,z):
    gen=policy.action_generator;prefix=gen.prepare_prefix(representation)
    n=representation.agentview_token_count;weight=representation.view_log_trust.exp()
    prefix=[replace(c,value=torch.cat([c.value[:,:,:n]*weight,c.value[:,:,n:]],dim=2)) for c in prefix]
    x=z.clone();vv=[]
    for k in range(gen.num_sample_steps):
        t=torch.full((len(z),),k/gen.num_sample_steps,device=x.device,dtype=x.dtype)
        v=gen(x,t,prefix);vv.append(v);x=x+v/gen.num_sample_steps
    return x,torch.stack(vv)


def energy(v,objective='relative',anchor_norm=None):
    # [time,noise,horizon,channel], each noise receives its own relative score.
    w=torch.ones(len(v),device=v.device,dtype=v.dtype)
    if objective=='late':w[6:]=2
    w=w/w.sum();w=w[:,None,None,None]
    mean=(w*v).sum(0);var=(w*(v-mean.unsqueeze(0)).square()).sum(0).mean((-1,-2))
    power=(w*v.square()).sum(0).mean((-1,-2));relative=var/power.clamp_min(1e-12)
    norm=mean.flatten(1).norm(dim=1)
    value=relative.mean()
    if objective=='anchored':
        if anchor_norm is None:raise ValueError('target-only initial speed anchor required')
        value=value+.1*(norm-anchor_norm).square().mean()
    return value,{'raw':var,'relative':relative,'mean_speed_norm':norm}


def build_representation(policy,pieces,delta,s,batch=1,original=False):
    ex=policy.representation_extractor
    if original:visual=ex.cross_attention_resampler(pieces['patches'],pieces['pool'])
    else:
        visual=resampler_forward_with_bias(ex.cross_attention_resampler,pieces['patches'],pieces['pool'],{'agentview':delta})
        n=visual.shape[1]//2
        if tuple(ex.view_names)!=('agentview','eye_in_hand'):raise ValueError('unexpected visual view order')
        # View trust is applied to post-LayerNorm prefix Values in integrate().
    tokens=torch.cat([visual,pieces['language'].to(visual.dtype),pieces['state']],dim=1)
    return SimpleNamespace(tokens=tokens.expand(batch,-1,-1),pooled_language=pieces['pool'].expand(batch,-1),action_dtype=tokens.dtype,view_log_trust=s,agentview_token_count=visual.shape[1]//2)


def project_(delta,s):
    with torch.no_grad():
        # Fix softmax gauge: at least one unsuppressed patch, all biases <=0.
        delta.sub_(delta.max()).clamp_(-8,0)
        s.clamp_(-4.605170186,0)  # trust in [0.01,1], still permits near-collapse tests.


def random_null(delta,s,generator):
    # Preserve per-block L2 norms. A negative scalar has no alternative direction.
    target=delta.norm()
    if float(target)<=1e-12:return torch.zeros_like(delta),s.detach().clone()
    u=torch.abs(torch.randn(delta.shape,device=delta.device,dtype=delta.dtype,generator=generator))
    u.view(-1)[u.argmin()]=0
    lo=0.;hi=1.
    while torch.minimum(u*hi,torch.full_like(u,8)).norm()<target:hi*=2
    for _ in range(60):
        mid=(lo+hi)/2
        if torch.minimum(u*mid,torch.full_like(u,8)).norm()<target:lo=mid
        else:hi=mid
    random_delta=-torch.clamp(u*((lo+hi)/2),max=8)
    return random_delta,s.detach().clone()


def optimize(policy,pieces,z,knob,objective,lr,steps=10):
    """No reference actions, images, outcomes, or Clean statistics accepted."""
    device=z.device;n=pieces['patches']['agentview'].shape[1]
    delta=torch.zeros((1,n),device=device,requires_grad=knob in ('spatial','joint'))
    s=torch.zeros((),device=device,requires_grad=knob in ('view','joint'))
    params=[p for p in (delta,s) if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=lr)
    with torch.no_grad():
        _,v=integrate(policy,build_representation(policy,pieces,delta,s,len(z)),z)
        _,b=energy(v);anchor=b['mean_speed_norm'].detach()
    snapshots={0:(delta.detach().clone(),s.detach().clone())};history=[]
    for k in range(1,steps+1):
        optimizer.zero_grad(set_to_none=True)
        _,v=integrate(policy,build_representation(policy,pieces,delta,s,len(z)),z)
        loss,stats=energy(v,objective,anchor)
        reg=.001*(delta.square().sum()+s.square());total=loss+reg
        if not torch.isfinite(total):raise ValueError('nonfinite objective')
        total.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('invalid gate gradient')
        history.append({'step':k,'pre_update_objective':float(loss.detach()),'regularizer':float(reg.detach()),
                        'raw_variance':float(stats['raw'].detach().mean()),'relative_variance':float(stats['relative'].detach().mean())})
        optimizer.step();project_(delta,s)
        if k in (5,10):snapshots[k]=(delta.detach().clone(),s.detach().clone())
    return snapshots,history
