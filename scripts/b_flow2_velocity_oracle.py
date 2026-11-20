"""B-Flow-2: per-token trace and bounded Euler correction at multiple rho."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch

ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from corp import load_policy  # noqa
from corp.preprocessing import prepare_observation  # noqa
from corp.representation.extractor import Representation  # noqa
from dacsc.action_metrics import action_error  # noqa


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=10); ap.add_argument('--rhos',default='0.05,0.1,0.15,0.2'); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); man=json.loads(Path(str(args.pairs)+'.manifest.json').read_text()); z=np.load(args.pairs); dev=torch.device(args.device); policy=load_policy(args.checkpoint,device=dev); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    ext=policy.representation_extractor; ag=policy.action_generator; W=ag.velocity_projection.weight.detach().cpu().numpy(); Wp=np.linalg.pinv(W); Wt=torch.tensor(W,device=dev,dtype=torch.float32); dt=1.0/args.steps; rhos=[float(x) for x in args.rhos.split(',')]
    def encoded(idx,domain):
        imgs={'agentview':torch.tensor(z[f'{domain}_agentview'][idx],device=dev,dtype=torch.uint8).unsqueeze(0),'eye_in_hand':torch.tensor(z[f'{domain}_eye'][idx],device=dev,dtype=torch.uint8).unsqueeze(0)}
        state=torch.tensor(z['state8'][idx],device=dev,dtype=torch.float32)[None]; prepared=prepare_observation({'images':imgs,'state':state,'instruction':[man['language']]},config=policy.config,normalizer=policy.normalizer,device=dev)
        vis=ext.visual_encoder(torch.cat([prepared['images'][v] for v in ext.view_names],dim=0)); patches={v:vis[j:j+1] for j,v in enumerate(ext.view_names)}; lang,pool=ext.language_encoder(prepared['instruction']); vt=ext.cross_attention_resampler(patches,pool); st=ext.state_encoder(prepared['state']); return Representation(tokens=torch.cat([vt,lang,st],dim=1),pooled_language=pool,action_dtype=vt.dtype)
    def hv(prefix,x,tau):
        t=torch.full((x.shape[0],),tau,device=dev,dtype=x.dtype); xin=ag.action_projection(x)+ag.action_position[:,:x.shape[1]]+ag.time_embedding(t).unsqueeze(1)
        for block,cache in zip(ag.blocks,prefix): xin=block(xin,cache)
        h=ag.output_norm(xin); return h,ag.velocity_projection(h)
    out=[]
    for idx,rec in enumerate(man['records']):
        if not rec['pairing_pass']: continue
        Rc=encoded(idx,'clean'); Rs=encoded(idx,'shift'); pc=ag.prepare_prefix(Rc); ps=ag.prepare_prefix(Rs); z0=torch.tensor(z['z0'][idx],device=dev,dtype=torch.float32)[None]
        xc=z0.clone(); xs=z0.clone(); xcorr=z0.clone(); traces=[]
        for n in range(args.steps):
            tau=n*dt; hc,vc=hv(pc,xc,tau); hs,vs=hv(ps,xs,tau); hcc,vcc=hv(pc,xcorr,tau); hsc,vsc=hv(ps,xcorr,tau)
            r=(vcc-vsc)[0].detach().cpu().numpy(); dh=(Wp@r.T).T; hn=np.linalg.norm(hsc[0].detach().cpu().numpy(),axis=1); dhn=np.linalg.norm(dh,axis=1); ratio=dhn/(hn+1e-12)
            for k in range(12): traces.append({'flow_step':n,'tau':tau,'k':k,'h_norm':float(hn[k]),'dh_norm':float(dhn[k]),'r_norm':float(np.linalg.norm(r[k])),'ratio':float(ratio[k])})
            # unbounded corrected branch
            xcorr=xcorr+(vsc+(torch.tensor(dh,device=dev,dtype=vsc.dtype)[None]@Wt.T))*dt
            xc=xc+vc*dt; xs=xs+vs*dt
        # bounded branches; recompute from z0 with per-step clamp at common corrected state
        bounded={}
        for rho in rhos:
            xb=z0.clone()
            for n in range(args.steps):
                tau=n*dt; hcb,vcb=hv(pc,xb,tau); hsb,vsb=hv(ps,xb,tau); r=(vcb-vsb)[0].detach().cpu().numpy(); dh=(Wp@r.T).T; hn=np.linalg.norm(hsb[0].detach().cpu().numpy(),axis=1); dhn=np.linalg.norm(dh,axis=1); scale=np.minimum(1.0,(rho*hn)/(dhn+1e-12)); dhb=dh*scale[:,None]
                xb=xb+(vsb+(torch.tensor(dhb,device=dev,dtype=vsb.dtype)[None]@Wt.T))*dt
            bounded[str(rho)]=action_error(policy.normalizer.unnormalize_action(xb)[0].detach().cpu().numpy(),policy.normalizer.unnormalize_action(xc)[0].detach().cpu().numpy(),executed=True,horizon=12)
        ac=policy.normalizer.unnormalize_action(xc)[0].detach().cpu().numpy(); ash=policy.normalizer.unnormalize_action(xs)[0].detach().cpu().numpy(); acorr=policy.normalizer.unnormalize_action(xcorr)[0].detach().cpu().numpy()
        out.append({'role':rec['role'],'state_id':rec['state_id'],'traces':traces,'shift':action_error(ash,ac,executed=True,horizon=12),'bounded':bounded,'unbounded_parity':float(np.max(np.abs(acorr-ac)))})
        print(json.dumps({'state':rec['state_id'],'role':rec['role'],'max_ratio':float(max(t['ratio'] for t in traces))}),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'b_flow2_velocity_oracle_v1','steps':args.steps,'rhos':rhos,'records':out},indent=2)+'\n')
    # summary by rho
    summ={}
    for rho in rhos:
        summ[str(rho)]={k:float(np.mean([r['bounded'][str(rho)][k] for r in out])) for k in ('xyz_norm_mean','xyz_m_mean','rotation_rad_geodesic_mean','gripper_abs_mean')}
    alltr=[t for r in out for t in r['traces']]; byflow={str(n):float(np.mean([t['ratio'] for t in alltr if t['flow_step']==n])) for n in range(args.steps)}; byk={str(k):float(np.mean([t['ratio'] for t in alltr if t['k']==k])) for k in range(12)}
    print(json.dumps({'n':len(out),'rhos':summ,'by_flow':byflow,'by_action_position':byk,'parity':max(r['unbounded_parity'] for r in out)},indent=2))

if __name__=='__main__': main()
