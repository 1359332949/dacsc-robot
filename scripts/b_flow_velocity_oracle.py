"""B-Flow: exact Euler trajectory, per-token B1, and Oracle integration parity."""
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
    ap.add_argument('--steps',type=int,default=10); ap.add_argument('--rho',type=float,default=0.1); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); man=json.loads(Path(str(args.pairs)+'.manifest.json').read_text()); z=np.load(args.pairs); dev=torch.device(args.device); policy=load_policy(args.checkpoint,device=dev); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    ext=policy.representation_extractor; ag=policy.action_generator; W=ag.velocity_projection.weight.detach().cpu().numpy(); Wp=np.linalg.pinv(W); dt=1.0/args.steps
    def encoded(idx,domain):
        imgs={'agentview':torch.tensor(z[f'{domain}_agentview'][idx],device=dev,dtype=torch.uint8).unsqueeze(0),'eye_in_hand':torch.tensor(z[f'{domain}_eye'][idx],device=dev,dtype=torch.uint8).unsqueeze(0)}
        state=torch.tensor(z['state8'][idx],device=dev,dtype=torch.float32)[None]; prepared=prepare_observation({'images':imgs,'state':state,'instruction':[man['language']]},config=policy.config,normalizer=policy.normalizer,device=dev)
        vis=ext.visual_encoder(torch.cat([prepared['images'][v] for v in ext.view_names],dim=0)); patches={v:vis[j:j+1] for j,v in enumerate(ext.view_names)}; lang,pool=ext.language_encoder(prepared['instruction']); vt=ext.cross_attention_resampler(patches,pool); st=ext.state_encoder(prepared['state']); R=Representation(tokens=torch.cat([vt,lang,st],dim=1),pooled_language=pool,action_dtype=vt.dtype); return R,vt,st
    def hv(prefix,x,tau):
        t=torch.full((x.shape[0],),tau,device=dev,dtype=x.dtype)
        xin=ag.action_projection(x)+ag.action_position[:,:x.shape[1]]+ag.time_embedding(t).unsqueeze(1)
        for block,cache in zip(ag.blocks,prefix): xin=block(xin,cache)
        h=ag.output_norm(xin); return h,ag.velocity_projection(h)
    out=[]
    for idx,rec in enumerate(man['records']):
        if not rec['pairing_pass']: continue
        Rc,vtc,_=encoded(idx,'clean'); Rs,vts,_=encoded(idx,'shift'); pc=ag.prepare_prefix(Rc); ps=ag.prepare_prefix(Rs); z0=torch.tensor(z['z0'][idx],device=dev,dtype=torch.float32)[None]
        xc=z0.clone(); xs=z0.clone(); xcorr=z0.clone(); ratios=[]; max_ratio=0
        for n in range(args.steps):
            tau=n*dt; hc,vc=hv(pc,xc,tau); hs,vs=hv(ps,xs,tau)
            # common state for B1/parity: corrected branch state (equals clean trajectory)
            hc_common,vc_common=hv(pc,xcorr,tau); hs_common,vs_common=hv(ps,xcorr,tau)
            r=(vc_common-vs_common)[0].detach().cpu().numpy(); dh=(Wp@r.T).T
            ratio=np.linalg.norm(dh,axis=1)/(np.linalg.norm(hs_common[0].detach().cpu().numpy(),axis=1)+1e-12); ratios.extend(ratio.tolist()); max_ratio=max(max_ratio,float(ratio.max()))
            xc=xc+vc*dt; xs=xs+vs*dt
            xcorr=xcorr+(vs_common+(torch.tensor(dh,device=dev,dtype=vs_common.dtype)[None]@torch.tensor(W,device=dev,dtype=vs_common.dtype).T))*dt
        # final actions
        norm_c=policy.normalizer.unnormalize_action(xc); norm_s=policy.normalizer.unnormalize_action(xs); norm_corr=policy.normalizer.unnormalize_action(xcorr)
        ac=norm_c[0].detach().cpu().numpy(); ash=norm_s[0].detach().cpu().numpy(); acorr=norm_corr[0].detach().cpu().numpy()
        e_shift=action_error(ash,ac,executed=True,horizon=12); e_corr=action_error(acorr,ac,executed=True,horizon=12)
        out.append({'role':rec['role'],'state_id':rec['state_id'],'ratio_mean':float(np.mean(ratios)),'ratio_max':max_ratio,'ratio_p95':float(np.quantile(ratios,.95)),'rho':args.rho,'ratio_ok':bool(max_ratio<=args.rho),
                    'final_shift':e_shift,'final_corrected':e_corr,'oracle_final_max_abs_diff':float(np.max(np.abs(acorr-ac)))})
        print(json.dumps({'state':rec['state_id'],'role':rec['role'],'ratio_max':max_ratio,'parity':out[-1]['oracle_final_max_abs_diff']}),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'b_flow_velocity_oracle_v1','steps':args.steps,'rho':args.rho,'records':out},indent=2)+'\n')
    print(json.dumps({'n':len(out),'ratio_ok':sum(r['ratio_ok'] for r in out),'ratio_max_mean':float(np.mean([r['ratio_max'] for r in out])),'parity_max':float(np.max([r['oracle_final_max_abs_diff'] for r in out])),'shift_xyz_m':float(np.mean([r['final_shift']['xyz_m_mean'] for r in out])),'corr_xyz_m':float(np.mean([r['final_corrected']['xyz_m_mean'] for r in out]))},indent=2))

if __name__=='__main__': main()
