"""Offline pairing: missing/duplicate, gate and replay checks before success metrics."""
import argparse,json,math
from pathlib import Path
from collections import Counter
from statistics import median

METHODS=('rgb','trust','joint')

def transitions(ref, meth, ids):
    b=sum(not ref[i]['success'] and meth[i]['success'] for i in ids);c=sum(ref[i]['success'] and not meth[i]['success'] for i in ids);d=b+c
    p=min(1.,2*sum(math.comb(d,k) for k in range(min(b,c)+1))/2**d) if d else 1.
    return dict(n=len(ids),frozen=sum(ref[i]['success'] for i in ids),success=sum(meth[i]['success'] for i in ids),rescued=b,harmed=c,net=b-c,exact_p=p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seed',default=0,type=int);ap.add_argument('--methods',nargs='+',default=list(METHODS));a=ap.parse_args()
    refs=json.loads(a.baseline.read_text());ref={r['case_id']:r for r in refs['rows']};assert len(ref)==len(refs['rows'])==100 and refs['seed_index']==a.seed
    historical=ref
    fr=[json.loads(p.read_text()) for p in a.raw.glob(f'run_s{a.seed}_*_frozen.json')];ref={r['case_id']:r for r in fr}
    assert len(fr)==100 and set(ref)==set(historical)
    clean=json.loads((a.raw/'clean_moments.json').read_text());assert len(clean)==len({r['base_task'] for r in clean})==10 and all(r['initial_sha']==ref[r['case_id']]['initial_sha'] for r in clean)
    arms={};records={}
    for method in a.methods:
        rows=[json.loads(p.read_text()) for p in a.raw.glob(f'run_s{a.seed}_*_{method}.json')];rr={r['case_id']:r for r in rows};assert len(rows)==len(rr)==100 and set(rr)==set(ref)
        records[method]=rr
        for cid,r in rr.items():
            b=ref[cid];assert r['seed']==a.seed and r['method']==method and r['factor']==b['factor'] and r['initial_sha']==b['initial_sha'] and r['wrist_sha']==b['wrist_sha']
            assert len(r['action_sha'])==64 and 0<r['steps']<=220 and r['trace'][0]['state']==b['trace'][0]['state']
            assert [q['step'] for q in r['trace']]==list(range(0,r['steps'],12))
            assert all(0<n<=64 and 0<mx<=1 for _,_,n,mx in r['roundings'])
            if method!='rgb':
                g=r['gate'];assert g['accepted']==(g['validation_before']>1e-12 and g['validation_ratio']<=.8);assert len(set(g['training_noise_hashes']+g['validation_noise_hashes']))==8
                assert len(g['history'])==10 and (g['applied_trust']==1 if not g['accepted'] else .00999<=g['applied_trust']<=1)
                if not g['accepted'] and method=='trust':assert (r['action_sha'],r['success'],r['steps'])==(b['action_sha'],b['success'],b['steps']),(cid,'rollback differs from Frozen')
        by={}
        for group in ['all','clean','lighting','background','sensor_noise']:
            ids=[i for i,r in ref.items() if group=='all' or r['factor']==group];entry=transitions(ref,rr,ids)
            if method!='rgb':
                g=[rr[i]['gate'] for i in ids];entry.update(accepted=sum(x['accepted'] for x in g),accepted_original_success=sum(rr[i]['gate']['accepted'] and ref[i]['success'] for i in ids),median_applied_trust=median(x['applied_trust'] for x in g))
            by[group]=entry
        for grouping in ['base_task','sub_dimension']:
            by[grouping]={}
            def group_key(r):
                if grouping=='sub_dimension':return 'clean' if r['factor']=='clean' else r['factor']+'/'+str(r.get(grouping))
                return str(r.get(grouping))
            for val in sorted({group_key(x) for x in rr.values()}):
                ids=[i for i in rr if group_key(rr[i])==val];by[grouping][val]=transitions(ref,rr,ids)
        arms[method]=by
    joint_checks=0
    if {'rgb','joint'}<=records.keys():
        for i,r in records['joint'].items():
            b=records['rgb'][i];assert r['wrist_sha']==b['wrist_sha'];
            if not r['gate']['accepted']:
                assert (r['action_sha'],r['success'],r['steps'])==(b['action_sha'],b['success'],b['steps']),(i,'joint rollback mismatch');joint_checks+=1
    report=dict(seed=a.seed,methods=arms,qualifiers=[m for m,r in arms.items() if r['all']['net']>=6 and r['all']['success']>=79 and r['clean']['success']==10],joint_rollback_hashes_verified=joint_checks)
    report['frozen_vs_archive']=transitions(historical,ref,list(ref))
    report['archive_mismatches']=[dict(case_id=i,factor=r['factor'],pixels=r['initial_sha']!=historical[i]['init_agentview_sha256'],steps=r['steps']!=historical[i]['steps'],outcome=r['success']!=historical[i]['success']) for i,r in ref.items() if r['initial_sha']!=historical[i]['init_agentview_sha256'] or r['steps']!=historical[i]['steps'] or r['success']!=historical[i]['success']]
    report['render_corrections']={m:dict(episodes=sum(bool(r['roundings']) for r in rr.values()),views=sum(len(r['roundings']) for r in rr.values()),max_pixels=max((entry[2] for r in rr.values() for entry in r['roundings']),default=0),max_intensity=max((entry[3] for r in rr.values() for entry in r['roundings']),default=0)) for m,rr in [('frozen',ref),*records.items()]}
    report['frozen_matches_historical_by_factor']={f:transitions(historical,ref,[i for i,r in ref.items() if r['factor']==f]) for f in ('clean','sensor_noise','lighting','background')}
    report['compared_to_historical_for_context_only']={m:transitions(historical,rr,list(ref)) for m,rr in records.items()}
    report['crossover']={}
    report['trigger_breakdown']={}
    # Full set of missing-equivalence records must be empty, not silently ignored.
    assert not list(a.raw.glob(f'run_s{a.seed}_*.pending'))
    if 'joint' in records and 'trust' in records:
        for f in ('clean','sensor_noise','background','lighting'):
            pairs=[(records['trust'][i]['gate']['accepted'],records['joint'][i]['gate']['accepted']) for i in ref if ref[i]['factor']==f]
            report['crossover'][f]={'neither':pairs.count((False,False)),'raw_only':pairs.count((True,False)),'rgb_only':pairs.count((False,True)),'both':pairs.count((True,True))}
    for m,rr in records.items():
        if m=='rgb':continue
        report['trigger_breakdown'][m]={}
        for f in ('clean','sensor_noise','lighting','background'):
            ids=[i for i,r in rr.items() if r['factor']==f];gated=[i for i in ids if rr[i]['gate']['accepted']];refgroup=records['rgb'] if m=='joint' and 'rgb' in records else ref
            report['trigger_breakdown'][m][f]=dict(n=len(ids),accepted=len(gated),accepted_paired_transitions=transitions(refgroup,rr,gated),candidate_trusts=[rr[i]['gate']['candidate_trust'] for i in ids],accepted_trusts=[rr[i]['gate']['applied_trust'] for i in gated])
    if set(records)==set(METHODS):report['joint_vs_rgb']=transitions(records['rgb'],records['joint'],list(ref))
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
