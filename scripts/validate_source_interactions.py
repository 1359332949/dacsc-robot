"""Reopen Source Clean shards; verify full role coverage and prevent pool leakage."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.source_interactions import role_ids,validate_source_episode


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root,split_path,require_complete=True):
    split=json.loads(split_path.read_text());split_hash=sha(split_path)
    reports=[];seen=set();signatures=set()
    for path in sorted(root.glob('*/*/metadata.json')):
        m=json.loads(path.read_text());npz=path.parent/'episode.npz';key=(m['role'],m['state_id'],m['seed'])
        if key in seen:raise ValueError('duplicate source episode '+str(key))
        seen.add(key)
        if m['split_sha256']!=split_hash:raise ValueError('split fingerprint mismatch')
        if sha(npz)!=m['npz_sha256']:raise ValueError('NPZ hash mismatch')
        if m['checkpoint_sha256']!=split['policy']['checkpoint_sha256']:raise ValueError('checkpoint mismatch')
        with np.load(npz,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        result=validate_source_episode(a,m,split)
        if m['steps'] != result['executed_steps']:raise ValueError('metadata step count mismatch')
        if (m['H'],m['E'],m['max_steps'])!=(12,12,220):raise ValueError('source execution protocol mismatch')
        if not np.array_equal(a['truncated'],a['collector_truncated']|a['env_truncated']):raise ValueError('truncation provenance mismatch')
        if not m['official_sampler_parity']:raise ValueError('missing sampler check')
        for check in m['official_sampler_parity']:
            if any(check[k]>1e-5 for k in ('x_max_abs','velocity_max_abs','controller_max_abs','clipped_max_abs')):raise ValueError('sampler mismatch')
        if bool(a['success'][a['executed_mask']][-1])!=m['success']:raise ValueError('outcome mismatch')
        signatures.add((m['script_sha256'],m['source_contract_sha256'],m['transition_contract_sha256'],m['shared_inference_sha256'],m['precision'],m['renderer'],m['seed']))
        reports.append({'role':m['role'],'state_id':m['state_id'],'seed':m['seed'],'success':m['success'],'steps':m['steps'],
                        'validation':result,'directory':str(path.parent.relative_to(root)),
                        'npz_sha256':m['npz_sha256'],'metadata_sha256':sha(path)})
    if not reports:raise ValueError('no source episodes')
    if len(signatures)!=1:raise ValueError('mixed collection provenance')
    summaries={}
    for role in ('source_fit','source_validation'):
        rows=[r for r in reports if r['role']==role];observed=sorted(r['state_id'] for r in rows);expected=sorted(role_ids(split,role))
        if require_complete and observed!=expected:raise ValueError('incomplete '+role+': '+str(observed))
        summaries[role]={'states':observed,'expected_states':expected,'episodes':len(rows),'successes':sum(r['success'] for r in rows),
                         'steps':sum(r['steps'] for r in rows),'full_chunks':sum(r['validation']['full_chunks'] for r in rows),
                         'partial_chunks':sum(r['validation']['partial_chunks'] for r in rows),
                         'complete':observed==expected}
    return {'schema':'source_clean_dataset_validation_v1','pass':True,'coverage_complete':all(r['complete'] for r in summaries.values()),
            'split_sha256':split_hash,'root':str(root),'roles':summaries,'episodes':reports,
            'claim_scope':'independent Source Clean data; no observer training, threshold calibration, or risk performance claim',
            'outcome_policy':'all predeclared episodes retained; Clean does not imply successful or risk-free'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-incomplete',action='store_true');a=p.parse_args()
    d=validate(a.root,a.split,not a.allow_incomplete);a.out.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(json.dumps(d['roles'],indent=2))
