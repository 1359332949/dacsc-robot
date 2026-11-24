"""Oracle-gradient capacity probe at the pre-Resampler token interface.

This is a proxy for a small LayerNorm at DINO->Resampler, because raw images
are not stored in the patch-pair NPZ. It tests whether a trainable LayerNorm at
that interface can reduce Shift-Clean action error without Clean drift.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from corp import load_policy  # noqa
from corp.representation.extractor import Representation  # noqa
from dacsc.differentiable_policy import sample_from_z0  # noqa


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--manifest',type=Path,default=None); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=30); ap.add_argument('--batch',type=int,default=16); ap.add_argument('--lr',type=float,default=1e-2); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); man=json.loads((args.manifest or args.pairs.with_suffix('.manifest.json')).read_text()); recs=man['records']; z=np.load(args.pairs,allow_pickle=False)
    fit=[r for r in recs if r.get('role')=='source_fit' and r.get('pairing_pass')]; val=[r for r in recs if r.get('role')=='source_validation' and r.get('pairing_pass')]
    if not fit or not val: raise RuntimeError('missing split')
    policy=load_policy(args.checkpoint,device=torch.device(args.device)); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    rs=policy.representation_extractor.cross_attention_resampler
    class ResidualNorm(nn.Module):
        def __init__(self):
            super().__init__(); self.ln=nn.LayerNorm(384); self.residual_scale=nn.Parameter(torch.zeros(1))
        def forward(self,x): return x + self.residual_scale*self.ln(x)
    adapters=nn.ModuleDict({v:ResidualNorm().to(args.device) for v in ('agentview','eye_in_hand')})
    params=list(adapters.parameters())
    opt=torch.optim.Adam(params,lr=args.lr)
    def tensor(r,key,idx): return torch.tensor(z[key][idx],device=args.device,dtype=torch.float32)
    def batches(rows,bs):
        for i in range(0,len(rows),bs): yield rows[i:i+bs]
    def actions(rows,adapt=False):
        ac=[]; ash=[]; clean_adapt=[]
        for rs_rows in batches(rows,args.batch):
            idx=[int(r['feature_row']) for r in rs_rows]
            pc={v:tensor(None,f'clean_{v}',idx) for v in ('agentview','eye_in_hand')}
            ps={v:tensor(None,f'shift_{v}',idx) for v in ('agentview','eye_in_hand')}
            lang=tensor(None,'clean_language_tokens',idx); st=tensor(None,'clean_state_token',idx); pooled=tensor(None,'clean_pooled_language',idx); z0=tensor(None,'z0',idx)
            def run(patches):
                visual=rs(patches,pooled); rep=Representation(tokens=torch.cat([visual,lang.to(visual.dtype),st],dim=1),pooled_language=pooled,action_dtype=visual.dtype)
                return policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))
            clean=run(pc)
            if adapt:
                ps={v:adapters[v](ps[v]) for v in ps}; pc={v:adapters[v](pc[v]) for v in pc}
                clean_adapt.append(run(pc)); ash.append(run(ps))
            ac.append(clean)
        return torch.cat(ac), (torch.cat(ash) if adapt else None), (torch.cat(clean_adapt) if adapt else None)
    # cache frozen clean reference and initial shift/clean
    c_fit,_,_=actions(fit,False); c_val,_,_=actions(val,False)
    def oracle_loss(rows,cref):
        c,ash,cad=actions(rows,True)
        return ((ash-cref)**2).mean()
    init_fit=oracle_loss(fit,c_fit).item(); init_val=oracle_loss(val,c_val).item()
    for step in range(args.steps):
        opt.zero_grad(); loss=oracle_loss(fit,c_fit); loss.backward(); grad=sum(float(p.grad.norm()) for p in params if p.grad is not None); opt.step()
        if step%5==0: print(json.dumps({'step':step,'loss':loss.item(),'grad_norm':grad}))
    final_fit=oracle_loss(fit,c_fit).item(); final_val=oracle_loss(val,c_val).item()
    # clean drift with adapted clean
    with torch.no_grad():
        c,ash,cad=actions(val,True); drift=((cad-c_val)**2).mean().item(); shift=((ash-c_val)**2).mean().item()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'oracle_gradient_probe_v1','init_fit':init_fit,'final_fit':final_fit,'init_val':init_val,'final_val':final_val,'clean_drift_val':drift,'shift_error_val':shift,'steps':args.steps,'lr':args.lr,'params':sum(p.numel() for p in params)},indent=2)+'\n')
    print(json.dumps({'init_fit':init_fit,'final_fit':final_fit,'init_val':init_val,'final_val':final_val,'clean_drift_val':drift,'shift_error_val':shift},indent=2))

if __name__=='__main__': main()
