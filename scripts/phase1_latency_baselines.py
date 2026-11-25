"""Phase 1 latency/memory table + external-baseline self-check."""
from __future__ import annotations
import json,time,sys,os
from pathlib import Path
import numpy as np,torch
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
from corp import load_policy
from corp.preprocessing import prepare_observation
from dacsc.single_block_probe import align,score
from target_only_action_validation import sample_z,visual_from_obs
INPUT=ROOT/'runs/invariance_sensitivity_v1b/input'; OUT=ROOT/'runs/phase1'
ARMS=['frozen','v1_block3','pixel_align','deep_block12','actmad_proxy']

def raw_from_npz(path):
    z=np.load(path,allow_pickle=False); return {'images':{v:torch.from_numpy(z[v].copy())[None] for v in ('agentview','eye_in_hand')},'state':torch.from_numpy(z['state'].copy()),'instruction':['pick up the black bowl and place it on the plate']}
def pixel_stats(raws):
    st={}
    for v in ('agentview','eye_in_hand'):
        xs=torch.cat([r['images'][v] for r in raws],0).float()/255.
        st[v]=(xs.mean((0,1,2)),xs.std((0,1,2),unbiased=False))
    return st
def pixel_align_raw(raw,st):
    out={'state':raw['state'],'instruction':raw['instruction'],'images':{}}
    for v in ('agentview','eye_in_hand'):
        x=raw['images'][v].float()/255.; m=x.mean((1,2),keepdim=True); s=x.std((1,2),unbiased=False,keepdim=True)+1e-6; mu,ss=st[v]; y=((x-m)/s*ss+mu).clamp(0,1); out['images'][v]=(y*255).round().clamp(0,255).to(torch.uint8)
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True); policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False); ext=policy.representation_extractor
    rows=[]
    for f in sorted(INPUT.glob('index_*.json')): rows+=json.load(open(f))
    clean=[r for r in rows if r.get('factor')=='clean'][:10]; raws=[raw_from_npz(INPUT/(r['id']+'.npz')) for r in clean]; pst=pixel_stats(raws); raw=raws[0]; z=torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(0))
    full=np.load(ROOT/'runs/full_traj_v1/full.npz',allow_pickle=False); ref=tuple(torch.as_tensor(full[f"{clean[0]['task']}__3__{k}"],device='cuda:0').float() for k in ('mean','std','tau'))
    def setup(arm):
        h=None; holder={}
        if arm=='v1_block3': h=ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align(x,ref,.75,'dead'))
        elif arm=='deep_block12': h=ext.visual_encoder.backbone.blocks[11].register_forward_hook(lambda m,a,x:align(x,ref,.75,'dead'))
        elif arm=='actmad_proxy':
            scale=torch.ones((2,1,384),device='cuda:0',requires_grad=True); holder['scale']=scale
            def hook(m,a,x): return torch.cat((x[:,:1],x[:,1:]*scale),1)
            h=ext.visual_encoder.backbone.blocks[2].register_forward_hook(hook)
        return h
    def fwd(arm):
        r=pixel_align_raw(raw,pst) if arm=='pixel_align' else raw
        _,rep=visual_from_obs(policy,r); return sample_z(policy,rep,z)
    out={};
    for arm in ARMS:
        h=setup(arm)
        try:
            for _ in range(5): fwd(arm)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
            for _ in range(30): fwd(arm)
            torch.cuda.synchronize(); ms=(time.perf_counter()-t0)/30*1000; peak=torch.cuda.max_memory_allocated()/1024**2
            out[arm]={'forward_ms':ms,'peak_mem_mb':peak,'backward':0}
            if arm=='actmad_proxy':
                torch.cuda.synchronize(); t1=time.perf_counter(); scale=torch.ones((2,1,384),device='cuda:0',requires_grad=True)
                # one gradient update on the proxy scale, using block-3 stats
                cap={}
                def hh(m,a,x): cap['x']=x[:,1:].detach()*scale; return torch.cat((x[:,:1],cap['x']),1)
                h2=ext.visual_encoder.backbone.blocks[2].register_forward_hook(hh)
                try:
                    with torch.enable_grad():
                        batch=prepare_observation(raw,config=policy.config,normalizer=policy.normalizer,device='cuda:0'); ext.visual_encoder(torch.cat([batch['images'][v] for v in ext.view_names])); p=cap['x']; m=p.mean(1); s=p.std(1,unbiased=False); loss=score(m,s,ref[0],ref[1]).mean(); loss.backward(); torch.cuda.synchronize(); out[arm]['actmad_update_ms']=(time.perf_counter()-t1)*1000; out[arm]['actmad_loss']=float(loss.detach()); out[arm]['actmad_scale_changed']=float((scale.grad.abs().sum()).detach()); out[arm]['backward']=1
                finally: h2.remove()
        finally:
            if h: h.remove()
    # self checks
    checks={}
    def check(arm):
        raw2=pixel_align_raw(raw,pst) if arm=='pixel_align' else raw; h=setup(arm)
        try:
            with torch.no_grad(): _,rep=visual_from_obs(policy,raw2); a=sample_z(policy,rep,z); checks[arm]={'finite_action':bool(torch.isfinite(a).all()),'action_abs_mean':float(a.abs().mean())}
        finally:
            if h: h.remove()
    for arm in ARMS: check(arm)
    # pixel image diff
    r2=pixel_align_raw(raw,pst); checks['pixel_align']['image_mean_abs_diff']=float((r2['images']['agentview'].float()-raw['images']['agentview'].float()).abs().mean())
    # deep block12 finite only; actmad proxy update recorded
    OUTJS=OUT/'latency_memory.json'; OUTJS.write_text(json.dumps({'arms':out,'checks':checks,'device':torch.cuda.get_device_name(0)},indent=2)+'\n'); print(json.dumps({'arms':out,'checks':checks},indent=2))
if __name__=='__main__': main()
