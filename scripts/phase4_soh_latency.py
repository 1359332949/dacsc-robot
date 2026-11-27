"""Real CUDA-synchronized latency for the SOH v2 adapter (Clean vs Sensor)."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch
import phase3_smolvla_shift as base
import phase3_smolvla_unified_diag as diag
from dacsc.smolvla_soh import SOHModule, SOHModuleV2, SmolVLASOHAdapter
from phase3_smolvla_common import load_actor, reference_for, seed_torch

ROOT=Path('/hy-tmp/patch-diag'); DIAG=ROOT/'runs/phase3_smolvla_unified_diag_v1'; CALIB=ROOT/'runs/phase3_smolvla/spatial_init0_calib.npz'

def load_soh(path):
    p=torch.load(path,map_location='cuda',weights_only=False)
    cls=SOHModuleV2 if p.get('arch')=='v2' else SOHModule
    if 'v_basis' in p and 'eigenvalues' in p:
        m=cls(v_basis=p['v_basis'].cuda().float(),eigenvalues=p['eigenvalues'].cuda().float(),rank=int(p['rank']),num_bases=int(p['num_bases']),damping_beta=float(p.get('damping_beta',1.0))).cuda()
    else:
        m=cls(p['u_geo'].cuda().float(),rank=int(p['rank']),num_bases=int(p['num_bases'])).cuda()
    m.load_state_dict(p['state_dict']); return m.eval()

def main(a):
    m=json.loads((DIAG/'manifest.json').read_text()); gates=m['protocol']['gate']
    actor=load_actor(); soh=load_soh(a.soh); funcs=dict(base.corruption_functions(),glass_blur=diag.glass_blur_exact)
    obs={}
    for g in ('Clean','Sensor'):
        c=next(c for c in m['cases'] if c['group']==g)
        suite,_,raw=base.setup(c); obs[g]=(c,base.perturb(raw,c,0,funcs),reference_for(CALIB,c['task_id']),gates[str(c['task_id'])]['threshold']); suite.close()
    rows=[]
    for rnd in range(5):
        for g in ('Clean','Sensor'):
            c,o,refs,tau=obs[g]
            for arm in (('frozen','soh') if rnd%2==0 else ('soh','frozen')):
                actor.reset(0); seed_torch(0)
                ad=SmolVLASOHAdapter(actor._model,refs,tau,soh) if arm=='soh' else None
                try:
                    if ad: ad.begin()
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
                    action=actor.act(o)
                    torch.cuda.synchronize(); ms=1000*(time.perf_counter()-t)
                    if ad: ad.end()
                    assert np.isfinite(action).all()
                    rows.append(dict(group=g,arm=arm,round=rnd,ms=ms,peak_mib=torch.cuda.max_memory_allocated()/1024**2))
                finally:
                    if ad: ad.close()
    groups={}
    for g in ('Clean','Sensor'):
        groups[g]={}
        for arm in ('frozen','soh'):
            vals=[r['ms'] for r in rows if r['group']==g and r['arm']==arm]
            mem=[r['peak_mib'] for r in rows if r['group']==g and r['arm']==arm]
            groups[g][arm]=dict(median_ms=float(np.median(vals)),peak_mib=float(np.median(mem)),rows=vals)
        groups[g]['relative']=groups[g]['soh']['median_ms']/groups[g]['frozen']['median_ms']
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(schema='phase4_soh_latency_v1',rounds=5,rows=rows,groups=groups,
        scope='end-to-end native actor.act including CPU preprocessing, device transfer, gate and SOH; synchronized CUDA; adapter construction excluded'),indent=2)+'\n')
    print(json.dumps(groups,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--soh',type=Path,required=True); p.add_argument('--out',type=Path,required=True); main(p.parse_args())
