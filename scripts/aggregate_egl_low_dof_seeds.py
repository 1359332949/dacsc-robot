"""Repeat seeds reuse the same cases. Task-cluster interval is exploratory."""
import argparse, json
from pathlib import Path
import numpy as np
from collections import defaultdict

def aggregate(dirs):
    rows=[];ids=None
    for seed,d in enumerate(dirs):
        read=lambda m: {x['case_id']:x for p in d.glob(f'run_s{seed}_*_{m}.json') for x in [json.loads(p.read_text())]}
        b,v=read('frozen'),read('trust');assert len(b)==len(v)==100 and set(b)==set(v)
        if ids is not None:assert set(b)==ids
        ids=set(b)
        for i in sorted(b):
            x,y=b[i],v[i];rows.append(dict(seed=seed,case_id=i,task=x['base_task'],factor=x['factor'],subtype=x['sub_dimension'] if x['factor']!='clean' else 'clean',baseline=int(x['success']),method=int(y['success']),delta=int(y['success'])-int(x['success']),accepted=y['gate']['accepted'],applied=y['gate']['applied_trust']))
    def stats(rr):
        tasks=sorted({r['task'] for r in rr});total=len(rr);msg={}
        for k in ('baseline','method'):msg[k+'_successes']=sum(r[k] for r in rr)
        msg.update(n=total,cases=len({r['case_id'] for r in rr}),tasks=len(tasks),recovered=sum(r['delta']==1 for r in rr),harmed=sum(r['delta']==-1 for r in rr),net=sum(r['delta'] for r in rr),gain_pp=100*np.mean([r['delta'] for r in rr]),accepted=sum(r['accepted'] for r in rr))
        rng=np.random.default_rng(20261124);draw=rng.integers(len(tasks),size=(5000,len(tasks)));cnt=np.array([sum(r['task']==t for r in rr) for t in tasks]);wins=np.array([sum(r['delta'] for r in rr if r['task']==t) for t in tasks]);interval=100*wins[draw].sum(1)/cnt[draw].sum(1);msg['task_bootstrap_gain_pp_95']=np.quantile(interval,[.025,.975]).tolist();return msg
    flip=defaultdict(list)
    for r in rows:flip[r['case_id']].append(r['accepted'])
    return {'groups':{f:stats([r for r in rows if f=='all' or r['factor']==f]) for f in ['all','sensor_noise','clean','background','lighting']},'sensor_subtypes':{t:stats([r for r in rows if r['factor']=='sensor_noise' and r['subtype']==t]) for t in sorted({r['subtype'] for r in rows if r['factor']=='sensor_noise'})},'task_net':{t:sum(r['delta'] for r in rows if r['task']==t) for t in sorted({r['task'] for r in rows})},'gate_changed_across_seeds':sum(len(set(x))>1 for x in flip.values()),'rows':rows}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',nargs=3,type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.write_text(json.dumps(aggregate(a.raw),indent=2,allow_nan=False)+'\n')
