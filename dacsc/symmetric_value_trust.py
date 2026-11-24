"""Projected two-view trust; gate inputs are the frozen representation and a noise seed."""
from dataclasses import replace
import math
import torch
from dacsc.static_visual_v2 import array_hash, seed_for, sensor_transform
from dacsc.flow_value_trust import with_trust,accept_update
from dacsc.flow_repair_kv import energy

LOW=-4.605170186

def project_two(u):
    if u.shape!=(2,) or not u.isfinite().all():raise ValueError('invalid two-view update')
    y=u.double();v=y.clamp(LOW,0);gain=2*y*v-v.square()  # reduction in squared distance from identity
    if gain[0]==gain[1]:return torch.zeros_like(u)
    keep=int(gain.argmax());ret=torch.zeros_like(u);ret[keep]=v[keep].to(u.dtype);return ret


def gate_values(cache,s,n):
    if s.shape!=(2,) or n<1:raise ValueError('expected one scalar per view')
    w=s.exp();out=[]
    for c in cache:
        v=c.value
        if v.shape[2]<2*n:raise ValueError('prefix too short')
        out.append(replace(c,value=torch.cat((v[:,:,:n]*w[0],v[:,:,n:2*n]*w[1],v[:,:,2*n:]),dim=2)))
    return out


def solve(gen,cache,z,s,n):
    p=gate_values(cache,s,n);x=z.clone();vs=[]
    for k in range(gen.num_sample_steps):
        tau=torch.full((len(z),),k/gen.num_sample_steps,dtype=z.dtype,device=z.device)
        v=gen(x,tau,p);x=x+v/gen.num_sample_steps;vs.append(v)
    return x,torch.stack(vs)


def symmetric_fit(policy,base,seed):
    device=base.tokens.device;gen=policy.action_generator;n=base.agentview_token_count
    def noise(ns):
        zs=[torch.randn((1,12,7),device=device,generator=torch.Generator(device=device).manual_seed(seed_for(seed,'flow_value_closed_loop_v1',ns,i))) for i in range(4)]
        return torch.cat(zs),[array_hash(z.cpu().numpy()) for z in zs]
    fit,h1=noise('fit');val,h2=noise('validation');assert len(set(h1+h2))==8
    s=torch.zeros(2,device=device,requires_grad=True);optimizer=torch.optim.Adam([s],lr=.3)
    with torch.no_grad():cache=gen.prepare_prefix(with_trust(base,s[0],4));before=float(energy(solve(gen,cache,val,s,n)[1])[0])
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True);loss=energy(solve(gen,cache,fit,s,n)[1])[0]+.001*s.square().sum();loss.backward()
        if not torch.isfinite(loss) or not s.grad.isfinite().all():raise ValueError('nonfinite fit')
        optimizer.step()
        with torch.no_grad():s.copy_(project_two(s))
    with torch.no_grad():after=float(energy(solve(gen,cache,val,s,n)[1])[0]);candidate=s.tolist()
    accept=accept_update(before,after);logs=candidate if accept else [0.,0.];eye=('agent' if logs[0]<0 else 'wrist') if min(logs)<0 else 'none'
    return dict(accepted=accept,candidate=candidate,logs=logs,eye=eye,before=before,after=after,train_hash=h1,val_hash=h2)


def perturb(obs,code,seed,step,wrapper,wrist):
    # Evaluator-only camera routing; the fitting/inference functions never receive wrist.
    if not wrist:return sensor_transform(obs,code,seed,step,wrapper)
    key='robot0_eye_in_hand_image';temp=dict(obs,agentview_image=obs[key]);image=sensor_transform(temp,code,seed,step,wrapper)['agentview_image']
    return dict(obs,robot0_eye_in_hand_image=image)
