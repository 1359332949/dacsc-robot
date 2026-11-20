"""Gate 3: B0 h-only ridge inside full 10-step bounded Euler."""
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
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--ridge',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--steps',type=int,default=10); ap.add_argument('--rho',type=float,default=.1); ap.add_argument('--device',default='cuda:0'); args=ap.parse_args()
    man=json.loads(Path(str(args.pairs)+'.manifest.json').read_text()); z=np.load(args.pairs); dev=torch.device(args.device); policy=load_policy(args.checkpoint,device=dev); policy.eval(); R=json.loads(args.ridge.read_text()); beta=np.array(R['beta']); mu=np.array(R['mu']); sd=np.array(R['sd']); ext=policy.representation_extractor; ag=policy.action_generator; W=ag.velocity_projection.weight.detach().cpu().numpy(); Wp=np.linalg.pinv(W); dt=1/args.steps
    def enc(idx,dom):
        imgs={'agentview':torch.tensor(z[f'{dom}_agentview'][idx],device=dev,dtype=torch.uint8).unsqueeze(0),'eye_in_hand':torch.tensor(z[f'{dom}_eye'][idx],device=dev,dtype=torch.uint8).unsqueeze(0)}; st=torch.tensor(z['state8'][idx],device=dev,dtype=torch.float32)[None]; pr=prepare_observation({'images':imgs,'state':st,'instruction':[man['language']]},config=policy.config,normalizer=policy.normalizer,device=dev); vis=ext.visual_encoder(torch.cat([pr['images'][v] for v in ext.view_names],dim=0)); patches={v:vis[j:j+1] for j,v in enumerate(ext.view_names)}; lang,pool=ext.language_encoder(pr['instruction']); vt=ext.cross_attention_resampler(patches,pool); s=ext.state_encoder(pr['state']); return Representation(tokens=torch.cat([vt,lang,s],dim=1),pooled_language=pool,action_dtype=vt.dtype),st[0]
    def hv(prefix,x,tau):
        t=torch.full((x.shape[0],),tau,device=dev,dtype=x.dtype); a=ag.action_projection(x)+ag.action_position[:,:x.shape[1]]+ag.time_embedding(t).unsqueeze(1)
        for b,c in zip(ag.blocks,prefix): a=b(a,c)
        h=ag.output_norm(a); return h,ag.velocity_projection(h)
    out=[]
    for idx,rec in enumerate(man['records']):
        if not rec['pairing_pass']: continue
        Rc,S=enc(idx,'clean'); Rs,_=enc(idx,'shift'); pc=ag.prepare_prefix(Rc); ps=ag.prepare_prefix(Rs); z0=torch.tensor(z['z0'][idx],device=dev,dtype=torch.float32)[None]; xc=z0.clone(); xs=z0.clone(); xb=z0.clone(); ratios=[]
        for n in range(args.steps):
            tau=n*dt; hc,vc=hv(pc,xc,tau); hs,vs=hv(ps,xs,tau); hb,vb=hv(ps,xb,tau); feat=np.r_[hb[0,0].detach().cpu().numpy(), S.detach().cpu().numpy(), tau, 0.0]; r=(torch.tensor(((feat-mu)/sd),device=dev,dtype=vb.dtype,requires_grad=False)[None]@torch.tensor(beta[:-1],device=dev,dtype=vb.dtype)); # beta linear incl? 
            # use full affine: beta has intercept last
            r=(torch.tensor(((feat-mu)/sd),device=dev,dtype=vb.dtype)[None]@torch.tensor(beta[:-1],device=dev,dtype=vb.dtype))+torch.tensor(beta[-1],device=dev,dtype=vb.dtype)
            dh=(Wp@r[0].detach().cpu().numpy().reshape(7,1)).reshape(-1); dh=dh*np.minimum(1.0,(args.rho*np.linalg.norm(hb[0,0].detach().cpu().numpy()))/(np.linalg.norm(dh)+1e-12)); corr=torch.tensor(dh,device=dev,dtype=vb.dtype)
            xb=xb+(vb+(torch.tensor(W,device=dev,dtype=vb.dtype)@corr)[None,None,:])*dt; xs=xs+vs*dt; xc=xc+vc*dt
        ac=policy.normalizer.unnormalize_action(xc)[0].detach().cpu().numpy(); ash=policy.normalizer.unnormalize_action(xs)[0].detach().cpu().numpy(); ab=policy.normalizer.unnormalize_action(xb)[0].detach().cpu().numpy(); out.append({'role':rec['role'],'state_id':rec['state_id'],'shift':action_error(ash,ac,executed=True,horizon=12),'b0':action_error(ab,ac,executed=True,horizon=12)}); print(json.dumps({'state':rec['state_id'],'shift_xyz':out[-1]['shift']['xyz_m_mean'],'b0_xyz':out[-1]['b0']['xyz_m_mean']}),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'b0_bounded_integration_v1','rho':args.rho,'records':out},indent=2)+'\n')

if __name__=='__main__': main()
