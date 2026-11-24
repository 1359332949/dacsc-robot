"""F-SFC t=0 offline probe. Default is spec only; real inference requires --execute."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.fsfc import ARMS, Config, VIEWS, fit_priors, frobenius, propose, source_ids, transform
from dacsc.flow_repair import energy, integrate
from dacsc.static_visual_v2 import array_hash, authorize, file_hash, seed_for

FINAL = ROOT / 'docs/data/static_visual_v2_suite/final'
SKETCHES = ROOT / 'docs/data/global_shift_diag_v1/feature_sketches.tar.gz'
CHECKPOINT = 'f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6'


def perturbed_images(images):
    """One deterministic representative per family; [2,H,W,3] uint8 RGB."""
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError('expected uint8 RGB')
    x=images.float(); mean=x.mean((1,2),keepdim=True)
    yy=torch.arange(x.shape[1],device=x.device)[:,None]
    xx=torch.arange(x.shape[2],device=x.device)[None,:]
    def blur(kernel):
        k=torch.tensor(kernel,device=x.device,dtype=x.dtype);k=k/k.sum();h,w=k.shape
        y=F.pad(x.permute(0,3,1,2),(w//2,w//2,h//2,h//2),mode='reflect')
        return F.conv2d(y,k.expand(3,1,h,w),groups=3).permute(0,2,3,1)
    for name,y in [('brightness',x*.8),('contrast',mean+.8*(x-mean)),
                   ('temperature',x*x.new_tensor([1.12,1.,.88])),('gamma',255*(x/255).pow(1.25)),
                   ('texture',x+8*(2*((yy+xx)%2)-1)[None,:,:,None]),
                   ('gaussian',blur([[1,2,1],[2,4,2],[1,2,1]])),('motion',blur([[1]*5]))]:
        yield name,y.round().clamp(0,255).to(torch.uint8)


def draws(seed,kind,device):
    return torch.cat([torch.randn((1,12,7),device=device,generator=torch.Generator(device=device).manual_seed(
        seed_for(seed,'fsfc_v1',kind,j))) for j in range(4)])


def binding(policy, state, language, rgb):
    from corp.preprocessing import prepare_observation
    device=next(policy.parameters()).device;ex=policy.representation_extractor
    if tuple(ex.view_names)!=VIEWS:raise ValueError('camera order drift')
    def prepare(images):
        po={'images':{v:images[i:i+1] for i,v in enumerate(VIEWS)},'state':state,'instruction':[language]}
        return prepare_observation(po,config=policy.config,normalizer=policy.normalizer,device=device)
    with torch.no_grad():
        b=prepare(rgb); lang,pool=ex.language_encoder(b['instruction']);st=ex.state_encoder(b['state'])
    @torch.no_grad()
    def patches(images):
        b=prepare(images)
        return ex.visual_encoder(torch.cat([b['images'][v] for v in VIEWS]))
    def forward(p,z):
        vis=ex.cross_attention_resampler({v:p[i:i+1] for i,v in enumerate(VIEWS)},pool)
        tokens=torch.cat((vis,lang.to(vis.dtype),st),1)
        rep=SimpleNamespace(tokens=tokens.expand(len(z),-1,-1),pooled_language=pool.expand(len(z),-1),action_dtype=vis.dtype)
        x,v=integrate(policy,rep,z)
        return policy.normalizer.unnormalize_action(x),v
    return patches,forward


def execute(a):
    if not all((a.features,a.corp_src,a.checkpoint,a.out)):raise ValueError('execution requires cache, CoRP, checkpoint and output paths')
    manifest=FINAL/'dev_manifest.json'; mh=file_hash(manifest);m=json.loads(manifest.read_text()); cases=m['cases']
    expected=json.loads((FINAL/'manifest_hashes.json').read_text())[manifest.name]
    if m['kind']!='dev' or m['status']!='FROZEN' or len(cases)!=100 or mh!=expected or mh!='d3e0d8c642cd215c879d2f0174663ff5ec585cd5c6d6b531a213c26e9fb982c0':raise ValueError('expected frozen Dev 100')
    by={c['case_id']:c for c in cases}
    for c in cases:authorize(m,c,'fsfc offline t0',expected,manifest)
    with tarfile.open(SKETCHES) as t:
        index={cid:json.load(t.extractfile('features/'+cid+'.json')) for cid in by}
    selected=[cid for cid,r in index.items() if r['factor']=='clean' or not r['baseline_success']]
    if len(selected)!=42:raise ValueError('selection drift: expected 10 Clean + 32 historical failures')
    if file_hash(a.checkpoint)!=CHECKPOINT:raise ValueError('checkpoint drift')
    sys.path.insert(0,str(a.corp_src));from corp import load_policy
    policy=load_policy(a.checkpoint,device='cuda:0').eval().requires_grad_(False)
    if policy.action_generator.num_sample_steps!=10:raise ValueError('wrong flow sampler')
    a.out.mkdir(parents=True,exist_ok=False); cache={}; folds={}; per_source={}
    def load(cid):
        if cid in cache:return cache[cid]
        c=by[cid];r=index[cid];path=a.features/(cid+'.npz');rp=FINAL/'results'/(cid+'.json')
        if (r['manifest_sha256']!=mh or r['case_id']!=cid or r['base_task']!=c['base_task'] or r['factor']!=c['factor'] or
                r['feature_npz_sha256']!=file_hash(path) or r['baseline_result_sha256']!=file_hash(rp)):raise ValueError('unverified cache')
        ref=json.loads(rp.read_text());seed=ref['seeds']['method_augmentation_seed']
        with np.load(path,allow_pickle=False) as arr:
            rgb=torch.tensor(np.stack([arr['rgb_'+v] for v in VIEWS]),device='cuda:0'); state=torch.tensor(arr['state'])
            p=torch.tensor(np.stack([arr['dino_'+v] for v in VIEWS]),device='cuda:0'); z=torch.tensor(arr['z0'],device='cuda:0')
            if [array_hash(x.cpu().numpy()) for x in z.split(1)]!=r['z0_hashes']:raise ValueError('z0 drift')
            enc,forward=binding(policy,state,ref['language_instruction'],rgb)
            with torch.no_grad():
                if p.shape!=(2,256,384) or (enc(rgb)-p).abs().max()>1e-5:raise ValueError('DINO replay mismatch')
                frozen=forward(p,z)[0]; recorded=torch.as_tensor(arr['actions'],device=z.device)
                if not torch.isfinite(frozen).all() or (frozen-recorded).abs().max()>1e-4:raise ValueError('identity action mismatch')
        cache[cid]=(p,z,frozen.detach(),seed,enc,forward,rgb);return cache[cid]
    def fold(task):
        if task not in folds:
            ids=source_ids(cases,task)
            for cid in ids:
                if cid in per_source:continue
                p,z,_,_,enc,forward,rgb=load(cid)
                drift=torch.stack([(enc(y)-p).abs().mean(1) for _,y in perturbed_images(rgb)]).mean(0)
                imp=frobenius(lambda x,n:forward(x,n)[0],p,z)
                per_source[cid]=(p,drift,imp)
            priors=fit_priors(*(torch.stack(xs) for xs in zip(*(per_source[i] for i in ids))))
            folds[task]=(ids,priors)
            # Retain the numeric denominator and every ablation weight for review.
            record={'held_task':task,'source_ids':ids,'mean':priors['uniform'].mean.tolist(),
                    'scale':priors['uniform'].scale.tolist(),'rho':{k:v.rho.tolist() for k,v in priors.items()},
                    'mean_action_jacobian':torch.stack([per_source[i][2] for i in ids]).mean(0).tolist(),
                    'mean_abs_perturbation':torch.stack([per_source[i][1] for i in ids]).mean(0).tolist()}
            directory=a.out/'priors'; directory.mkdir(exist_ok=True)
            (directory/(hashlib.sha256(task.encode()).hexdigest()[:16]+'.json')).write_text(json.dumps(record,allow_nan=False)+'\n')
        return folds[task]
    clean={c['pair_group_id']:c['case_id'] for c in cases if c['role']=='dev_clean'}
    for cid in selected:
        c=by[cid];ids,priors=fold(c['base_task']);p,z,base,seed,_,forward,_=load(cid)
        fit,valid=(draws(seed,k,p.device) for k in ('fit','acceptance'))
        if set(index[cid]['z0_hashes']) & {array_hash(x.cpu().numpy()) for n in (fit,valid) for x in n.split(1)}:raise ValueError('evaluation noise reuse')
        def curvature(q,n):return energy(forward(q,n)[1])[0]
        arms={}
        for arm in ARMS:
            g,info=propose(p,priors[arm],fit,valid,curvature)
            with torch.no_grad():actions=forward(transform(p,g,priors[arm]),z)[0]
            arms[arm]={'decision':info,'gate':g.cpu().tolist(),'actions':actions.cpu().tolist()}
        # Privileged reference opened for scoring only AFTER all arms have finished.
        ref_id=clean[c['pair_group_id']];ref=load(ref_id)[2]
        if index[cid]['z0_hashes']!=index[ref_id]['z0_hashes']:raise ValueError('unmatched oracle noise')
        for record in arms.values():
            diff=torch.tensor(record['actions'],device=p.device).clamp(-1,1)-ref.clamp(-1,1)
            record['reference_L2']={name:float(diff[...,sl].flatten(1).norm(dim=1).mean()) for name,sl in [('xyz',slice(0,3)),('rot',slice(3,6)),('all',slice(None))]}
        out={'case_id':cid,'source_case_ids':ids,'frozen_actions':base.cpu().tolist(),'paired_clean_reference':ref.cpu().tolist(),
             'arms':arms,'settings':asdict(Config()),'manifest_sha256':mh,'checkpoint_sha256':CHECKPOINT,'selection':'retrospective Dev diagnostic'}
        (a.out/(cid+'.json')).write_text(json.dumps(out,allow_nan=False)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    for arg in ('features','corp-src','checkpoint','out'):ap.add_argument('--'+arg,type=Path)
    args=ap.parse_args()
    if args.execute:execute(args)
    else:print(json.dumps({'mode':'spec_only','experiment_started':False,'arms':ARMS,'config':asdict(Config())}))
