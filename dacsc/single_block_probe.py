"""Forward-only single-block candidate (no domain condition in the operator)."""
import json
from pathlib import Path
import numpy as np
import torch
from dacsc.static_visual_v2 import array_hash,file_hash,seed_for

LAYERS=(3,6,9,12); VIEWS=('agentview','eye_in_hand');ALPHAS=(.25,.5,.75)

def moments(x):return x.mean(1),x.square().mean(1)

def fit(raw):
    m,q=raw.unbind(1);mu=m.mean(0);return mu,(q.mean(0)-mu.square()).clamp_min(0).sqrt()

def score(m,sd,mu,ss):return ((m-mu).square()+(sd-ss).square())/(ss.square()+1e-6)

def reference(raw):
    if raw.shape[0]!=9 or raw.shape[1]!=2:raise ValueError('expect nine independent scene sketches')
    mu,ss=fit(raw);inner=[]
    for i in range(9):
        other=torch.cat([raw[:i],raw[i+1:]]);m,q=raw[i];cm,cs=fit(other);inner.append(score(m,(q-m.square()).clamp_min(0).sqrt(),cm,cs))
    return tuple(x.float() for x in (mu,ss,torch.stack(inner).quantile(.95,dim=0)))

def align(x,ref,alpha,gate):
    if alpha==0:return x
    if alpha not in ALPHAS or gate not in ('uniform','dead'):raise ValueError('unplanned variant')
    mu,ss,tau=ref;patch=x[:,1:];m=patch.mean(1);sd=patch.std(1,unbiased=False)
    active=score(m,sd,mu,ss)>tau if gate=='dead' else torch.ones_like(m,dtype=torch.bool)
    gain=(((1-alpha)*sd+alpha*ss)/(sd.square()+1e-6).sqrt()).clamp(.5,2)
    # Explicit bypass for inactive channels avoids cancellation/reconstruction roundoff.
    p=torch.where(active[:,None,:],gain[:,None,:]*(patch-m[:,None,:])+((1-alpha)*m+alpha*mu)[:,None,:],patch)
    return torch.cat((x[:,:1],p),dim=1)

def run(a,cases,policy):
    from corp.preprocessing import prepare_observation
    from corp.representation.extractor import Representation
    from dacsc.flow_repair import integrate,energy
    from unittest.mock import patch
    inp=a.out/'input';out=a.out/'outputs';out.mkdir(exist_ok=True);rows=[r for p in sorted(inp.glob('index_*.json')) for r in json.loads(p.read_text())];by={r['id']:r for r in rows}
    original=[r for r in rows if not r['geometry']];assert len(rows)==len(by) and len(original)==100 and {r['id'] for r in original}=={c['case_id'] for c in cases}
    clean=[r for r in original if r['factor']=='clean'];pairs={r['task']:r for r in clean};assert len(clean)==len(pairs)==10
    # No peeking at the current task's Clean in fitting a reference, including tau.
    for r in original:
        c=pairs[r['task']];assert (r['sim'],r['state'],r['lang'])==(c['sim'],c['state'],c['lang']),('unpaired state',r['id'])
    ex=policy.representation_extractor;blocks=ex.visual_encoder.backbone.blocks;assert len(blocks)==12 and tuple(ex.view_names)==VIEWS
    def load(r):
        f=inp/(r['id']+'.npz');assert file_hash(f)==r['file_sha'];
        with np.load(f,allow_pickle=False) as z:return dict(images={v:torch.from_numpy(z[v].copy())[None] for v in VIEWS},state=torch.from_numpy(z['state'].copy()),instruction=[r['lang']])
    def prepare(r):return prepare_observation(load(r),config=policy.config,normalizer=policy.normalizer,device='cuda:0')
    def vis(b):return ex.visual_encoder(torch.cat([b['images'][v] for v in VIEWS]))
    sketches={};refs={}; groups={}
    for c in clean:
        stored={}
        def capture(l):
            def hook(mod,args,o):stored[l]=torch.stack(moments(o[:,1:].double()));return None
            return hook
        hooks=[blocks[l-1].register_forward_hook(capture(l)) for l in LAYERS]
        try:vis(prepare(c))
        finally:
            for h in hooks:h.remove()
        assert set(stored)==set(LAYERS);sketches[c['task']]=stored
    for t in pairs:
        tasks=sorted(set(pairs)-{t});assert len(tasks)==9;groups[t]=[pairs[k]['id'] for k in tasks]
        for l in LAYERS:refs[t,l]=reference(torch.stack([sketches[k][l] for k in tasks]))
    if a.shard==0:
        np.savez_compressed(a.out/'references.npz',**{f'{t}__{l}__{n}':x.cpu().numpy() for (t,l),vv in refs.items() for n,x in zip(('mean','std','tau'),vv)})
        (a.out/'sources.json').write_text(json.dumps(groups,indent=2)+'\n')
    for r in sorted(rows,key=lambda x:x['id'])[a.shard::a.workers]:
        t=r['task'];dest=out/(r['id']+'.npz');assert not dest.exists();po=load(r);b=prepare(r);lang,pool=ex.language_encoder(b['instruction']);state=ex.state_encoder(b['state']);z=torch.cat([torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(seed_for(20261124,'isp',t,j))) for j in range(4)])
        def predict():
            p=vis(b);visual=ex.cross_attention_resampler({v:p[i:i+1] for i,v in enumerate(VIEWS)},pool);tokens=torch.cat([visual,lang.to(visual.dtype),state],1);rep=Representation(tokens=tokens.expand(4,-1,-1),pooled_language=pool.expand(4,-1),action_dtype=visual.dtype);x,v=integrate(policy,rep,z);return policy.normalizer.unnormalize_action(x),energy(v)[1]['relative']
        base,be=predict();
        with patch('torch.randn',side_effect=lambda *args,**kw:z[:1].clone()):off=policy.predict(po)
        parity=float((base[:1]-off).abs().max());assert parity<=1e-4
        actions=[base];energies=[be];keys=['frozen'];checks=[]
        for l in LAYERS:
            for mode,alpha in [('uniform',0.)]+[(m,v) for m in ('uniform','dead') for v in ALPHAS]:
                h=blocks[l-1].register_forward_hook(lambda mod,args,o:align(o,refs[t,l],alpha,mode))
                try:act,en=predict()
                finally:h.remove()
                assert torch.isfinite(act).all() and torch.isfinite(en).all()
                if not alpha:
                    assert torch.equal(act,base) and torch.equal(en,be);checks.append(l);continue
                keys.append(f'{l}_{mode}_{alpha:g}');actions.append(act);energies.append(en)
        np.savez_compressed(dest,keys=keys,actions=torch.stack(actions).cpu().numpy(),energy=torch.stack(energies).cpu().numpy(),noise_hashes=[array_hash(x.cpu().numpy()) for x in z.split(1)],parity=parity,identity=checks)
        print('probed',r['id'],flush=True)
