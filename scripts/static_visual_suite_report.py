"""Summarize V2 runtime evidence without opening Blind or selecting models."""
from pathlib import Path
import argparse,collections,csv,json

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--audit-dir',default='p0_suite_context_fixed');ap.add_argument('--result-dir',default='source_smoke_context_fixed');a=ap.parse_args()
    audit=json.loads((a.root/a.audit_dir/'p0_audit.json').read_text());manifest=json.loads((a.root/a.audit_dir/'dev_manifest.json').read_text())
    rows=[json.loads(p.read_text()) for p in sorted((a.root/a.result_dir).glob('*.json'))]
    if any(r['role']!='source_fit' for r in rows):raise ValueError('P0 report cannot mix roles')
    by={r['case_id']:r for r in rows}
    if len(by)!=len(rows):raise ValueError('duplicate results')
    table=[]
    for task in audit['candidate_tasks']:
        ids=[r['case_id'] for r in manifest['cases'] if r['base_task']==task]
        rr=[by[i] for i in ids if i in by]
        table.append({'base_task':task,'planned':len(ids),'completed':sum(r['status']=='completed' for r in rr),'exceptions':sum(r['status']!='completed' for r in rr),'missing':len(ids)-len(rr),'success':sum(r.get('success',False) for r in rr),
                      'steps':sum(r.get('steps',0) for r in rr)})
    complete=len(rows)==len(manifest['cases']) and all(r['status']=='completed' for r in rows)
    zero=[r['base_task'] for r in table if r['completed']==5 and r['success']==0]
    report={'status':'COMPATIBILITY_NOT_PASSED' if zero or not complete else 'COMPATIBILITY_PASSED',
            'complete':complete,'planned':len(manifest['cases']),'recorded':len(rows),'successful':sum(r.get('success',False) for r in rows),'tasks':table,'zero_of_five_tasks':zero,
            'all_initial_physics_pair_checks_pass':all(r.get('physical_pair',{}).get('raw_state_equal') and r.get('physical_pair',{}).get('model_physics_compatible') for r in rows),
            'max_official_sampler_parity_error':max((p[k] for r in rows for c in r.get('chunks',[]) for p in [c.get('official_parity',{})] if p for k in ('x_max_abs','velocity_max_abs','controller_max_abs','clipped_max_abs')),default=None),
            'target_dev_runs':0,'blind_runs':0,'training_coverage':'unknown','note':'empirical five-state compatibility only; no statistical certification or task deletion'}
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'clean_smoke_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (a.out/'clean_smoke_tasks.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    print(json.dumps(report,indent=2))
