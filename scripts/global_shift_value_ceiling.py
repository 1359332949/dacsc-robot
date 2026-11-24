"""Matched-coverage oracle: replace BOTH visual Values, preserve all Keys."""
import argparse,json,sys,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from dacsc.static_visual_v2 import file_hash,authorize

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ['manifest','features','corp-src','checkpoint','out']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();m=json.loads(a.manifest.read_text());cases={c['case_id']:c for c in m['cases']};h=json.loads((a.manifest.parent/'manifest_hashes.json').read_text())[a.manifest.name]
    for c in m['cases']:authorize(m,c,'privileged both-view Value ceiling',h,a.manifest)
    rr=[json.loads(p.read_text()) for p in a.features.glob('*.json')]
    if len(rr)!=100 or any(r['status']!='completed' for r in rr):raise ValueError('complete features required')
    clean={r['pair_group_id']:r for r in rr if r['factor']=='clean'}
    selected=[r for r in rr if r['factor']=='clean' or (r['factor'] in ('background','lighting') and not r['baseline_success'])]
    sys.path[:0]=[str(a.corp_src),str(a.corp_src/'scripts')];os.environ.setdefault('MUJOCO_GL','osmesa');os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
    import torch
    from corp import load_policy
    from corp.action_generator import PrefixCache
    from global_shift_diag_v1 import restore_cache,expand_cache,sample_cache
    from dacsc.flow_repair_kv import energy
    if file_hash(a.checkpoint)!='f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6':raise ValueError('checkpoint changed')
    policy=load_policy(a.checkpoint,device='cuda:0');policy.eval();policy.requires_grad_(False);rows=[]
    for r in selected:
        p=a.features/(r['case_id']+'.npz');cr=clean[r['pair_group_id']];cp=a.features/(cr['case_id']+'.npz')
        if file_hash(p)!=r['feature_npz_sha256'] or file_hash(cp)!=cr['feature_npz_sha256']:raise ValueError('feature hash drift')
        z=np.load(p);ref=np.load(cp);c=restore_cache(z);cc=restore_cache(ref)
        v=c[0].value.clone();v[:,:,:96]=cc[0].value[:,:,:96]
        with torch.no_grad():
            act,vel,_=sample_cache(policy,expand_cache([PrefixCache(c[0].key,v,c[0].adaln)]),torch.tensor(z['z0'],device='cuda:0'));_,e=energy(vel)
        rows.append({'case_id':r['case_id'],'name':'oracle_both_value','privileged_oracle':True,'actions':act.cpu().tolist(),'relative_curvature':e['relative'].cpu().tolist(),'feature_sha256':file_hash(p)})
    a.out.write_text(json.dumps({'rows':rows,'script_sha256':file_hash(Path(__file__)),'scope':'matched coverage privileged diagnostic only'},indent=2)+'\n');print('both-Value ceiling complete',len(rows))
