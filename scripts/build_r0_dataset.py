"""Build source R0 dataset: deployable features -> teacher velocity residual."""
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
    ap.add_argument('--z0-count',type=int,default=4); ap.add_argument('--steps',type=int,default=10); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); man=json.loads(Path(str(args.pairs)+'.manifest.json').read_text()); z=np.load(args.pairs); dev=torch.device(args.device); policy=load_policy(args.checkpoint,device=dev); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    ext=policy.representation_extractor; ag=policy.action_generator; dt=1.0/args.steps
    def encoded(idx,domain):
        imgs={'agentview':torch.tensor(z[f'{domain}_agentview'][idx],device=dev,dtype=torch.uint8).unsqueeze(0),'eye_in_hand':torch.tensor(z[f'{domain}_eye'][idx],device=dev,dtype=torch.uint8).unsqueeze(0)}
        state=torch.tensor(z['state8'][idx],device=dev,dtype=torch.float32)[None]; prepared=prepare_observation({'images':imgs,'state':state,'instruction':[man['language']]},config=policy.config,normalizer=policy.normalizer,device=dev)
        vis=ext.visual_encoder(torch.cat([prepared['images'][v] for v in ext.view_names],dim=0)); patches={v:vis[j:j+1] for j,v in enumerate(ext.view_names)}; lang,pool=ext.language_encoder(prepared['instruction']); vt=ext.cross_attention_resampler(patches,pool); st=ext.state_encoder(prepared['state']); R=Representation(tokens=torch.cat([vt,lang,st],dim=1),pooled_language=pool,action_dtype=vt.dtype); return R,vt
    def hv(prefix,x,tau):
        t=torch.full((x.shape[0],),tau,device=dev,dtype=x.dtype); xin=ag.action_projection(x)+ag.action_position[:,:x.shape[1]]+ag.time_embedding(t).unsqueeze(1)
        for block,cache in zip(ag.blocks,prefix): xin=block(xin,cache)
        h=ag.output_norm(xin); return h,ag.velocity_projection(h)
    feats=[]; labels=[]; roles=[]; states=[]; clean_flags=[]; hs=[]; vs=[]; vcs=[]
    for idx,rec in enumerate(man['records']):
        if not rec['pairing_pass']: continue
        Rc,vtc=encoded(idx,'clean'); Rs,vts=encoded(idx,'shift'); pc=ag.prepare_prefix(Rc); ps=ag.prepare_prefix(Rs); state8=torch.tensor(z['state8'][idx],device=dev,dtype=torch.float32)
        for zi in range(args.z0_count):
            g=torch.Generator(device=dev); g.manual_seed(900000+zi*100003+idx); z0=torch.randn((1,ag.action_horizon,ag.action_dim),device=dev,dtype=torch.float32,generator=g)
            # shift uncorrected trajectory; teacher at same x
            xs=z0.clone()
            for n in range(args.steps):
                tau=n*dt; hs_s,vs_s=hv(ps,xs,tau); hc_s,vc_s=hv(pc,xs,tau)
                for k in range(12):
                    feats.append(torch.cat([vts.mean(dim=1)[0],hs_s[0,k],state8,torch.tensor([tau,k/12.0],device=dev)]).detach().cpu().numpy())
                    labels.append((vc_s-vs_s)[0,k].detach().cpu().numpy()); roles.append(rec['role']); states.append(rec['state_id']); clean_flags.append(0); hs.append(hs_s[0,k].detach().cpu().numpy()); vs.append(vs_s[0,k].detach().cpu().numpy()); vcs.append(vc_s[0,k].detach().cpu().numpy())
                xs=xs+vs_s*dt
            # clean trajectory; teacher no-op zero
            xc=z0.clone()
            for n in range(args.steps):
                tau=n*dt; hc,vc=hv(pc,xc,tau)
                for k in range(12):
                    feats.append(torch.cat([vtc.mean(dim=1)[0],hc[0,k],state8,torch.tensor([tau,k/12.0],device=dev)]).detach().cpu().numpy())
                    labels.append(np.zeros(7,dtype=np.float32)); roles.append(rec['role']); states.append(rec['state_id']); clean_flags.append(1); hs.append(hc[0,k].detach().cpu().numpy()); vs.append(vc[0,k].detach().cpu().numpy()); vcs.append(vc[0,k].detach().cpu().numpy())
                xc=xc+vc*dt
        print(json.dumps({'state':rec['state_id'],'role':rec['role'],'n':len(labels)}),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.out,features=np.stack(feats),labels=np.stack(labels),roles=np.array(roles),states=np.array(states),clean=np.array(clean_flags),hs=np.stack(hs),vs=np.stack(vs),vcs=np.stack(vcs)); Path(str(args.out)+'.manifest.json').write_text(json.dumps({'language':man['language'],'n':len(labels),'feature_dim':int(feats[0].shape[0])},indent=2)+'\n')

if __name__=='__main__': main()
