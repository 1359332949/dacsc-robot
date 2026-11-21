"""Reopen saved paired data and independently check hashes and record contracts."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dacsc.interaction_records import validate_pair


def validate(root):
    summary=json.loads((root/'summary.json').read_text())
    reports=[]
    for saved in summary['reports']:
        folder=root/saved['pair_id'];arrays={};meta={}
        for domain in ('clean','shift'):
            path=folder/(domain+'.npz');meta[domain]=json.loads((folder/(domain+'.json')).read_text())
            if hashlib.sha256(path.read_bytes()).hexdigest()!=meta[domain]['npz_sha256']:raise ValueError('NPZ hash mismatch')
            with np.load(path,allow_pickle=False) as data:arrays[domain]={k:data[k] for k in data.files}
        result=validate_pair(arrays['clean'],arrays['shift'],meta['clean'],meta['shift'])
        controls=saved['probe_controls']
        if controls['repeat_max_abs']>1e-6:raise ValueError('probe repeat')
        if not saved['static_model_checks']['pass']:raise ValueError('model mismatch')
        if max(saved['physics_max_abs_by_field'].values())>1e-6:raise ValueError('physics mismatch')
        for value in controls['view_ownership'].values():
            if value['other_view_max_abs']>1e-5 or value['changed_view_max_abs']<=0:raise ValueError('view ownership')
        for item in saved['official_sampler_parity_clean']+[saved['official_sampler_parity_shift']]:
            for key in ('x_max_abs','velocity_max_abs','controller_max_abs','clipped_max_abs'):
                if item[key]>1e-5:raise ValueError('sampler parity')
        a=arrays['clean'];end=[]
        for i,n in enumerate(a['executed_steps']):
            end.append({'chunk':i,'executed_steps':int(n),**{k:bool(a[k][i,n-1]) for k in ('success','terminated','truncated','done')}})
        reports.append({'pair_id':saved['pair_id'],'contract':result,'chunk_end_flags':end,
                        'npz_sha256':{d:meta[d]['npz_sha256'] for d in meta}})
    return {'schema':'interaction_replay_reopened_validation_v1','pass':True,
            'summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
            'root':str(root),'reports':reports}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result=validate(a.root);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
