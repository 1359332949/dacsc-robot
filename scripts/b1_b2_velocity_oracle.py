"""B1/B2: bounded hidden and structured visual velocity-oracle audit."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch

ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from corp import load_policy  # noqa
from corp.preprocessing import prepare_observation  # noqa
from corp.representation.extractor import Representation  # noqa


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tau',type=float,default=0.5); ap.add_argument('--rho',type=float,default=0.1); ap.add_argument('--alpha-max',type=float,default=1.0); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); man=json.loads(Path(str(args.pairs)+'.manifest.json').read_text()); z=np.load(args.pairs); dev=torch.device(args.device); policy=load_policy(args.checkpoint,device=dev); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    ext=policy.representation_extractor; ag=policy.action_generator; W=ag.velocity_projection.weight.detach().cpu().numpy(); Wp=np.linalg.pinv(W); tau=torch.tensor([args.tau],device=dev)
    out=[]
    for rec in man['records']:
        if not rec['pairing_pass']: continue
        i=z.files and None; idx=man['records'].index(rec)
        def rep(domain):
            imgs={'agentview':torch.tensor(z[f'{domain}_agentview'][idx],device=dev,dtype=torch.uint8).unsqueeze(0),
                  'eye_in_hand':torch.tensor(z[f'{domain}_eye'][idx],device=dev,dtype=torch.uint8).unsqueeze(0)}
            state=torch.tensor(z['state8'][idx],device=dev,dtype=torch.float32)[None]; z0=torch.tensor(z['z0'][idx],device=dev,dtype=torch.float32)[None]
            prepared=prepare_observation({'images':imgs,'state':state,'instruction':[man['language']]},config=policy.config,normalizer=policy.normalizer,device=dev)
            vis=ext.visual_encoder(torch.cat([prepared['images'][v] for v in ext.view_names],dim=0)); B=1; patches={v:vis[j*B:(j+1)*B] for j,v in enumerate(ext.view_names)}
            lang,pooled=ext.language_encoder(prepared['instruction']); vt=ext.cross_attention_resampler(patches,pooled); st=ext.state_encoder(prepared['state']); R=Representation(tokens=torch.cat([vt,lang,st],dim=1),pooled_language=pooled,action_dtype=vt.dtype)
            return R, vt, z0
        Rc,vtc,z0=rep('clean'); Rs,vts,_=rep('shift'); prefix_c=ag.prepare_prefix(Rc); prefix_s=ag.prepare_prefix(Rs); action=z0
        def hv(prefix):
            x=ag.action_projection(action)+ag.action_position[:,:action.shape[1]]+ag.time_embedding(tau).unsqueeze(1)
            for block,cache in zip(ag.blocks,prefix): x=block(x,cache)
            h=ag.output_norm(x); return h, ag.velocity_projection(h)
        hc,vc=hv(prefix_c); hs,vs=hv(prefix_s)
        rt=(vc-vs).detach().cpu().numpy().reshape(12,7); rh=hc.detach().cpu().numpy().reshape(-1)
        # B1 per-token minimum-norm delta h
        dh=(Wp@rt.T).T  # [12,768]
        b1_ratio=float(np.linalg.norm(dh)/(np.linalg.norm(rh)+1e-12)); b1_ok=bool(b1_ratio<=args.rho)
        # B2 single bounded scalar along visual-token difference direction, projected by W
        x=(vtc-vts).mean(dim=1).detach().cpu().numpy().reshape(-1)  # [768]
        A=W@x  # [7]
        alpha=float(np.dot(np.tile(A,12),rt.reshape(-1))/(np.dot(np.tile(A,12),np.tile(A,12))+1e-12)); alpha=float(np.clip(alpha,-args.alpha_max,args.alpha_max))
        pred=np.tile(alpha*A,12).reshape(12,7); r2=rt-pred; b2_rel=float(np.linalg.norm(r2)/(np.linalg.norm(rt)+1e-12))
        out.append({'role':rec['role'],'state_id':rec['state_id'],'b1_ratio':b1_ratio,'b1_ok':b1_ok,'b2_alpha':alpha,'b2_relative_residual':b2_rel,'r_norm':float(np.linalg.norm(rt)),'h_norm':float(np.linalg.norm(rh))})
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'b1_b2_velocity_oracle_v1','tau':args.tau,'rho':args.rho,'alpha_max':args.alpha_max,'records':out},indent=2)+'\n')
    print(json.dumps({'n':len(out),'b1_ok':sum(r['b1_ok'] for r in out),'b1_ratio_mean':float(np.mean([r['b1_ratio'] for r in out])),'b2_rel_mean':float(np.mean([r['b2_relative_residual'] for r in out])),'b2_alpha_mean':float(np.mean([r['b2_alpha'] for r in out]))},indent=2))

if __name__=='__main__': main()
