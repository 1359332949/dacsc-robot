"""P2: real RGB DINO LayerNorm oracle feasibility."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch
from torch import nn

ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from corp import load_policy  # noqa
from corp.preprocessing import prepare_observation  # noqa
from corp.representation.extractor import Representation  # noqa
from dacsc.differentiable_policy import sample_from_z0  # noqa


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=20); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--batch',type=int,default=10); ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--block',type=int,default=0); ap.add_argument('--norm',default='norm1')
    args=ap.parse_args(); man=json.loads(Path(str(args.pairs)+'.manifest.json').read_text()); z=np.load(args.pairs)
    roles=np.array([r['role'] for r in man['records']]); fit=np.where(roles=='source_fit')[0]; val=np.where(roles=='source_validation')[0]
    policy=load_policy(args.checkpoint,device=torch.device(args.device)); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    target=getattr(policy.representation_extractor.visual_encoder.backbone.blocks[args.block],args.norm)
    for p in target.parameters(): p.requires_grad_(True)
    train_params=[p for p in target.parameters()]
    dev=torch.device(args.device)
    def batch(idx):
        idx=np.asarray(idx)
        imgs={'agentview':torch.tensor(z['clean_agentview'][idx],device=dev,dtype=torch.uint8),
              'eye_in_hand':torch.tensor(z['clean_eye'][idx],device=dev,dtype=torch.uint8)}
        imgs_s={'agentview':torch.tensor(z['shift_agentview'][idx],device=dev,dtype=torch.uint8),
                'eye_in_hand':torch.tensor(z['shift_eye'][idx],device=dev,dtype=torch.uint8)}
        state=torch.tensor(z['state8'][idx],device=dev,dtype=torch.float32); z0=torch.tensor(z['z0'][idx],device=dev,dtype=torch.float32)
        inst=[man['language']]*len(idx)
        return imgs,imgs_s,state,z0,inst
    def actions(imgs,state,z0,inst):
        prepared=prepare_observation({'images':imgs,'state':state,'instruction':inst},config=policy.config,normalizer=policy.normalizer,device=dev)
        ext=policy.representation_extractor; views=tuple(ext.view_names)
        vis=ext.visual_encoder(torch.cat([prepared['images'][v] for v in views],dim=0)); B=state.shape[0]
        patches={v:vis[i*B:(i+1)*B] for i,v in enumerate(views)}
        lang,pooled=ext.language_encoder(prepared['instruction']); vt=ext.cross_attention_resampler(patches,pooled); st=ext.state_encoder(prepared['state'])
        rep=Representation(tokens=torch.cat([vt,lang.to(vt.dtype),st],dim=1),pooled_language=pooled,action_dtype=vt.dtype)
        return policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))
    with torch.no_grad():
        imc,ims,st,z0,inst=batch(fit); ref_fit=actions(imc,st,z0,inst)
        imc,ims,st,z0,inst=batch(val); ref_val=actions(imc,st,z0,inst)
    opt=torch.optim.Adam(train_params,lr=args.lr)
    def loss_with(idx,ref):
        imc,ims,st,z0,inst=batch(idx); pred=actions(ims,st,z0,inst); return ((pred-ref)**2).mean()
    init_fit=float(loss_with(fit,ref_fit).detach()); init_val=float(loss_with(val,ref_val).detach())
    for step in range(args.steps):
        opt.zero_grad(); loss=loss_with(fit,ref_fit); loss.backward(); gn=sum(float(p.grad.norm()) for p in train_params if p.grad is not None); opt.step()
        if step%5==0: print(json.dumps({'step':step,'loss':float(loss.detach()),'grad_norm':gn}))
    with torch.no_grad():
        final_fit=float(loss_with(fit,ref_fit)); final_val=float(loss_with(val,ref_val))
        imc,ims,st,z0,inst=batch(val); predv=actions(ims,st,z0,inst); cleanv=actions(imc,st,z0,inst)
        drift=float(((cleanv-ref_val)**2).mean()); shift=float(((predv-ref_val)**2).mean())
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'oracle_dino_p2_v1','block':args.block,'norm':args.norm,'steps':args.steps,'lr':args.lr,'init_fit':init_fit,'init_val':init_val,'final_fit':final_fit,'final_val':final_val,'clean_drift_val':drift,'shift_error_val':shift,'trainable_params':sum(p.numel() for p in train_params)},indent=2)+'\n')
    print(json.dumps({'init_fit':init_fit,'init_val':init_val,'final_fit':final_fit,'final_val':final_val,'clean_drift_val':drift,'shift_error_val':shift,'params':sum(p.numel() for p in train_params)},indent=2))

if __name__=='__main__': main()
