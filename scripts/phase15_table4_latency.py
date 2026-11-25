"""Phase 1.5 Table 4: latency/memory for 6 arms, 5 interleaved rounds."""
from __future__ import annotations
import json,time,sys,os,statistics
from pathlib import Path
import numpy as np,torch
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
from corp import load_policy
from corp.preprocessing import prepare_observation
from dacsc.single_block_probe import align
from dacsc.baselines_tta import Calib,pixel_align_raw,ActMAD
from target_only_action_validation import sample_z,visual_from_obs
INPUT=ROOT/'runs/invariance_sensitivity_v1b/input'; OUT=ROOT/'runs/phase15'
ARMS=['frozen','v1_block3','pixel_align','deep_block12','actmad_true','consensus_n8']
def raw_from_npz(p):
    z=np.load(p,allow_pickle=False); return {'images':{v:torch.from_numpy(z[v].copy())[None] for v in ('agentview','eye_in_hand')},'state':torch.from_numpy(z['state'].copy()),'instruction':['pick up the black bowl and place it on the plate']}
def main():
    OUT.mkdir(parents=True,exist_ok=True); policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval(); ext=policy.representation_extractor; cal=Calib(ROOT/'runs/phase2_calib/spatial_shared_calib.npz')
    rows=[]
    for f in sorted(INPUT.glob('index_*.json')): rows+=json.load(open(f))
    r=next(x for x in rows if x.get('factor')=='clean'); raw=raw_from_npz(INPUT/(r['id']+'.npz')); z=torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(0)); zs=[torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(i)) for i in range(8)]
    def make(arm):
        if arm=='v1_block3': return [ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align(x,cal.ref(3),.75,'dead'))]
        if arm=='deep_block12': return [ext.visual_encoder.backbone.blocks[11].register_forward_hook(lambda m,a,x:align(x,cal.ref(11),.75,'dead'))]
        if arm=='actmad_true': return [ActMAD(policy,cal)]
        return []
    def one(arm,objs):
        if arm=='pixel_align': raw2=pixel_align_raw(raw,cal)
        else: raw2=raw
        if arm=='actmad_true':
            ada=objs[0]; batch=prepare_observation(raw2,config=policy.config,normalizer=policy.normalizer,device='cuda:0'); ada.update(batch)
            with torch.no_grad(): _,rep=visual_from_obs(policy,raw2); a=sample_z(policy,rep,z)
            return a,True
        if arm=='consensus_n8':
            with torch.no_grad():
                outs=[]
                for zz in zs:
                    _,rep=visual_from_obs(policy,raw2); outs.append(sample_z(policy,rep,zz))
                return torch.stack(outs).mean(0),False
        with torch.no_grad():
            _,rep=visual_from_obs(policy,raw2); return sample_z(policy,rep,z),False
    # warmup 50 frozen
    for _ in range(50): one('frozen',[])
    torch.cuda.synchronize(); res={a:[] for a in ARMS}; peaks={a:[] for a in ARMS}; backs={}
    for rnd in range(5):
        order=ARMS[:]; __import__('random').Random(100+rnd).shuffle(order)
        for arm in order:
            objs=make(arm); ada=objs[0] if arm=='actmad_true' else None
            try:
                if ada: ada.reset()
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
                for _ in range(30): one(arm,objs)
                torch.cuda.synchronize(); dt=(time.perf_counter()-t0)/30*1000
                res[arm].append(dt); peaks[arm].append(torch.cuda.max_memory_allocated()/1024**2); backs[arm]=1 if arm=='actmad_true' else 0
            finally:
                for o in objs:
                    if hasattr(o,'close'): o.close()
                    else: o.remove()
    out={}
    frozen_med=statistics.median(res['frozen'])
    for a in ARMS:
        xs=res[a]; q=statistics.quantiles(xs,n=4) if len(xs)>1 else [xs[0],xs[0],xs[0]]; out[a]={'median_ms':statistics.median(xs),'iqr_ms':q[2]-q[0],'samples_ms':xs,'xfrozen':statistics.median(xs)/frozen_med,'backward':backs[a],'peak_mem_mb':max(peaks[a])}
    (OUT/'table4_latency.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
