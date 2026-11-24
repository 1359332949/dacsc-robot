"""Recompute clipped distances/drift from paired actions; check roles, noise and rollback."""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ARMS=('jacobian','base_std','uniform','jacobian_hard_mask')
CHANNELS={'xyz':slice(3),'rotation':slice(3,6),'all':slice(None)}

def dist(a,b):
    return {k:float(np.linalg.norm((a-b)[...,s].reshape(len(a),-1),axis=1).mean()) for k,s in CHANNELS.items()}

def vector(aa):
    a=np.asarray(aa)
    if a.shape!=(4,12,7) or not np.isfinite(a).all():raise ValueError('invalid controller actions')
    return a.clip(-1,1)

def pack(v):
    return {'n':len(v),'mean':float(np.mean(v)) if v else None,'max':max(v) if v else None,'median':float(np.median(v)) if v else None}

def summarize(rr):
    ans={'n':len(rr),'accepted':sum(r['accepted'] for r in rr),'rollback':sum(not r['accepted'] for r in rr)}
    ans['acceptance_rate']=ans['accepted']/len(rr);ans['rollback_rate']=1-ans['acceptance_rate']
    for view in ['agent','wrist']: ans['mean_g_'+view]=float(np.mean([r['g_'+view] for r in rr]))
    for ch in CHANNELS:
        b=[r['before'][ch] for r in rr];a=[r['after'][ch] for r in rr]
        reductions=[100*(1-y/x) for x,y in zip(b,a) if x>1e-4]
        ans[ch]={'distance_before':pack(b),'distance_after':pack(a),'reduction_pct':pack(reductions),'drift':pack([r['drift'][ch] for r in rr]),'ratio_of_totals_reduction_pct':100*(1-sum(a)/sum(b)) if sum(b)>1e-4 else None}
    return ans

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    m=json.loads((a.baseline/'dev_manifest.json').read_text());cases={c['case_id']:c for c in m['cases']}; paths=list(a.results.glob('*.json'))
    if len(paths)!=100 or {p.stem for p in paths}!=set(cases):raise ValueError('incomplete campaign')
    details=defaultdict(list); clean={c['pair_group_id']:c['case_id'] for c in cases.values() if c['role']=='dev_clean'}
    for p in sorted(paths):
        r=json.loads(p.read_text());cid=p.stem;c=cases[cid];ref=json.loads((a.baseline/'results'/(cid+'.json')).read_text())
        assert r['case_id']==cid and set(r['arms'])==set(ARMS) and ref['status']=='completed'
        assert set(r['source_case_ids'])=={x['case_id'] for x in cases.values() if x['role']=='dev_clean' and x['base_task']!=c['base_task']}
        assert len(set(r['fit_z_sha256']+r['validation_z_sha256']+r['eval_z_sha256']))==12
        base=vector(r['frozen_actions']);twin=vector(r['paired_clean_reference'])
        tc=json.loads((a.results/(clean[c['pair_group_id']]+'.json')).read_text());assert tc['eval_z_sha256']==r['eval_z_sha256'] and np.array_equal(twin,vector(tc['frozen_actions']))
        for arm,v in r['arms'].items():
            g=np.array(v['gate']);assert g.shape==(2,384) and np.all((0<=g)&(g<=1));assert v['settings']['hard_mask']==(arm=='jacobian_hard_mask')
            decision=v['decision'];ok=decision['accepted']
            if 'validation_L0_C0_L1_C1' in decision:
                l0,c0,l1,c1=decision['validation_L0_C0_L1_C1'];assert ok==(l0>0 and l1<=.95*l0 and c1<=1.05*c0)
            after=vector(v['actions']);assert ok or (not g.any() and np.array_equal(after,base))
            details[arm].append(dict(case_id=cid,group='clean' if c['factor']=='clean' else c['factor']+('_success' if ref['success'] else '_failure'),accepted=ok,g_agent=float(g[0].mean()),g_wrist=float(g[1].mean()),before=dist(base,twin),after=dist(after,twin),drift=dist(after,base)))
    groups={arm:{g:summarize([r for r in rr if r['group']==g]) for g in sorted({r['group'] for r in rr})} for arm,rr in details.items()}
    success={arm:summarize([r for r in rr if r['group'].endswith('_success')]) for arm,rr in details.items()}
    p=groups['jacobian'];checks={f:bool(p[f+'_failure']['xyz']['reduction_pct']['mean']>0 and p[f+'_failure']['rotation']['reduction_pct']['mean']>=0 and p[f+'_success']['xyz']['reduction_pct']['mean']>=0) for f in ['sensor_noise','background','lighting']}
    a.out.write_text(json.dumps(dict(schema='fsfc100_offline_v1',groups=groups,successful_shift_pooled=success,case_details=details,primary_screen=checks),indent=2,allow_nan=False)+'\n');print(checks)
if __name__=='__main__':main()
