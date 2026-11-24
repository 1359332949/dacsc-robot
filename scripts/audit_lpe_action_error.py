"""Offline: does LPE reduce shift-clean action deviation on same observations?"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from corp import load_policy  # noqa
from corp.representation.extractor import Representation  # noqa
from dacsc.differentiable_policy import sample_from_z0  # noqa


def adapt_one(p,mean,eps):
    a=p.float(); n=a.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    f=1.0+eps*torch.tanh(torch.log(torch.tensor(mean,device=a.device)/(n+1e-6)))
    return (a*f).to(p.dtype)


def adapt(patches,means,eps):
    if isinstance(patches,dict): return {v:adapt_one(t,means[v],eps) for v,t in patches.items()}
    return adapt_one(patches,means,eps)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--manifest',type=Path,default=None); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--max-scale-delta',type=float,default=0.2); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); man=json.loads((args.manifest or args.pairs.with_suffix('.manifest.json')).read_text()); recs=man['records']; z=np.load(args.pairs,allow_pickle=False)
    idx=[int(r['feature_row']) for r in recs if r.get('role')=='source_fit' and r.get('pairing_pass')]
    if not idx: raise RuntimeError('no source_fit rows')
    policy=load_policy(args.checkpoint,device=torch.device(args.device)); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    rs=policy.representation_extractor.cross_attention_resampler
    means={v:float(np.linalg.norm(z[f'clean_{v}'][idx].astype(np.float32),axis=-1).mean()) for v in ('agentview','eye_in_hand')}
    def act(patches,lang,state,pooled,z0):
        visual=rs(patches,pooled); rep=Representation(tokens=torch.cat([visual,lang.to(visual.dtype),state],dim=1),pooled_language=pooled,action_dtype=visual.dtype)
        with torch.no_grad(): return policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))[0].detach().cpu().numpy()
    out=[]
    for i in idx:
        pcl={v:torch.tensor(z[f'clean_{v}'][i],device=args.device,dtype=torch.float32)[None] for v in ('agentview','eye_in_hand')}
        psh={v:torch.tensor(z[f'shift_{v}'][i],device=args.device,dtype=torch.float32)[None] for v in ('agentview','eye_in_hand')}
        lang=torch.tensor(z['clean_language_tokens'][i],device=args.device,dtype=torch.float32)[None]; state=torch.tensor(z['clean_state_token'][i],device=args.device,dtype=torch.float32)[None]; pooled=torch.tensor(z['clean_pooled_language'][i],device=args.device,dtype=torch.float32)[None]; z0=torch.tensor(z['z0'][i],device=args.device,dtype=torch.float32)[None]
        ac=act(pcl,lang,state,pooled,z0); ash=act(psh,lang,state,pooled,z0); alpe=act(adapt(psh,means,args.max_scale_delta),lang,state,pooled,z0)
        dS=ash-ac; dL=alpe-ac
        v={'role':recs[i].get('role'),'state_id':int(recs[i]['state_id']),'window':recs[i].get('window'),'chunk':recs[i].get('chunk_index'),'feature_row':i,
           'delta_shift_norm':float(np.sqrt(np.mean(dS**2))),'delta_lpe_norm':float(np.sqrt(np.mean(dL**2))),
           'delta_shift_xyz':float(np.sqrt(np.mean(dS[:3]**2))),'delta_lpe_xyz':float(np.sqrt(np.mean(dL[:3]**2))),
           'delta_shift_rot':float(np.sqrt(np.mean(dS[3:6]**2))),'delta_lpe_rot':float(np.sqrt(np.mean(dL[3:6]**2)))}
        out.append(v)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'lpe_action_error_v1','max_scale_delta':args.max_scale_delta,'means':means,'records':out},indent=2)+'\n')
    print(json.dumps({'n':len(out),'mean_shift_norm':float(np.mean([r['delta_shift_norm'] for r in out])),'mean_lpe_norm':float(np.mean([r['delta_lpe_norm'] for r in out])),'mean_shift_xyz':float(np.mean([r['delta_shift_xyz'] for r in out])),'mean_lpe_xyz':float(np.mean([r['delta_lpe_xyz'] for r in out]))},indent=2))

if __name__=='__main__': main()
