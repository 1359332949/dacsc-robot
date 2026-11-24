"""t=0 scalar Value trust, independent-noise acceptance, no Clean inputs."""
from types import SimpleNamespace
import math
import torch
from dacsc.flow_repair_kv import integrate,energy
from dacsc.static_visual_v2 import seed_for,array_hash

METHODS=('frozen','flow_gated','fixed_010','gated_fixed_010')


def accept_update(before,after):
    if not math.isfinite(before) or not math.isfinite(after):raise ValueError('nonfinite validation energy')
    return before>1e-12 and after<=.8*before


def applied_log_trust(method,gate):
    if method=='frozen':return 0.
    if method=='fixed_010':return math.log(.1)
    if method=='flow_gated':return gate['candidate_log_trust'] if gate['accepted'] else 0.
    if method=='gated_fixed_010':return math.log(.1) if gate['accepted'] else 0.
    raise ValueError('unknown method')


def with_trust(base,s,batch=1):
    return SimpleNamespace(tokens=base.tokens.expand(batch,-1,-1),pooled_language=base.pooled_language.expand(batch,-1),
        action_dtype=base.action_dtype,view_log_trust=s,agentview_token_count=base.agentview_token_count)


def encode_native(policy,po):
    from corp.preprocessing import prepare_observation
    ex=policy.representation_extractor
    if tuple(ex.view_names)!=('agentview','eye_in_hand'):raise ValueError('view order changed')
    with torch.no_grad():
        batch=prepare_observation(po,config=policy.config,normalizer=policy.normalizer,device='cuda:0')
        rep=ex(**batch)
    n=int(ex.cross_attention_resampler.queries.shape[1])
    if n!=48:raise ValueError('expected 48 agentview prefix tokens')
    return SimpleNamespace(tokens=rep.tokens.detach(),pooled_language=rep.pooled_language.detach(),action_dtype=rep.action_dtype,agentview_token_count=n)


def fit_gate(policy,base,augmentation_seed):
    """Only target representation + augmentation seed; no baseline outcomes or Clean."""
    device=base.tokens.device
    with torch.random.fork_rng(devices=[device.index or 0]):
        def noise(namespace):
            zs=[];hashes=[]
            for j in range(4):
                g=torch.Generator(device=device).manual_seed(seed_for(augmentation_seed,'flow_value_closed_loop_v1',namespace,j))
                z=torch.randn((1,12,7),device=device,dtype=torch.float32,generator=g);zs.append(z);hashes.append(array_hash(z.cpu().numpy()))
            return torch.cat(zs),hashes
        train,train_hash=noise('fit');val,val_hash=noise('validation')
        if set(train_hash)&set(val_hash):raise ValueError('training and validation noises overlap')
        s=torch.zeros((),device=device,requires_grad=True)
        optimizer=torch.optim.Adam([s],lr=.3);history=[]
        with torch.no_grad():
            _,v=integrate(policy,with_trust(base,s,4),val);before=float(energy(v)[0])
        for k in range(10):
            optimizer.zero_grad(set_to_none=True)
            _,v=integrate(policy,with_trust(base,s,4),train)
            loss,_=energy(v);total=loss+.001*s.square()
            if not torch.isfinite(total):raise ValueError('nonfinite fit loss')
            total.backward()
            if s.grad is None or not torch.isfinite(s.grad):raise ValueError('invalid scalar gradient')
            history.append({'step':k+1,'pre_loss':float(loss.detach()),'pre_log_trust':float(s.detach()),'gradient':float(s.grad.detach())})
            optimizer.step()
            with torch.no_grad():s.clamp_(-4.605170186,0)
        with torch.no_grad():
            _,v=integrate(policy,with_trust(base,s,4),val);after=float(energy(v)[0])
        candidate=float(s.detach());accepted=accept_update(before,after)
    return {'candidate_log_trust':candidate,'candidate_trust':math.exp(candidate),'accepted':bool(accepted),
        'applied_log_trust':candidate if accepted else 0.,'applied_trust':math.exp(candidate) if accepted else 1.,
        'validation_before':before,'validation_after':after,'validation_ratio':after/before if before>1e-12 else None,
        'training_noise_hashes':train_hash,'validation_noise_hashes':val_hash,'history':history,
        'validation_role':'update acceptance, not an independent final test'}
