"""Report frozen episode outcomes from recorded state/seed keys, never a hand list."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(clean_path, shift_path):
    docs={d:json.loads(p.read_text()) for d,p in [('clean',clean_path),('shift',shift_path)]}
    keyed={}
    for d,doc in docs.items():
        records=[r for r in doc['records'] if r.get('method')=='frozen']
        keyed[d]={}
        for r in records:
            key=(r['init_index'],r['seed'])
            if key in keyed[d]:raise ValueError('duplicate state/seed')
            keyed[d][key]=r
    if keyed['clean'].keys()!=keyed['shift'].keys():raise ValueError('unpaired episode keys')
    pairs=[]
    for (state,seed),c in sorted(keyed['clean'].items()):
        s=keyed['shift'][(state,seed)]
        if not c['z0_schedule_sha256']==s['z0_schedule_sha256']:raise ValueError('z0 schedule mismatch')
        pairs.append(dict(state_id=state,seed=seed,clean_success=c['success'],shift_success=s['success'],
                          clean_steps=c['steps'],shift_steps=s['steps'],z0_schedule_sha256=c['z0_schedule_sha256']))
    return {'schema':'lighting_episode_outcome_audit_v1','sources':{
        d:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for d,p in [('clean',clean_path),('shift',shift_path)]},
        'per_seed':{str(seed):{d:{'success':sum(r[d+'_success'] for r in pairs if r['seed']==seed),
                                'states':[r['state_id'] for r in pairs if r['seed']==seed and r[d+'_success']]}
                     for d in ('clean','shift')} for seed in sorted({r['seed'] for r in pairs})},
        'total':{d:sum(r[d+'_success'] for r in pairs) for d in ('clean','shift')},
        'episodes_per_domain':len(pairs),'pairs':pairs,
        'claim_scope':'10 initial states x 3 seeds; repeated screening episodes, not 30 independent states'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clean',type=Path,required=True);p.add_argument('--shift',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=audit(a.clean,a.shift);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'total':result['total'],'per_seed':result['per_seed']},indent=2))
