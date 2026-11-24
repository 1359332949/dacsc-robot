"""Post-grid fixed token-scaling mechanism check, never used to select hyperparameters."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_flow_causal_repair import evaluate
from dacsc.static_visual_v2 import file_hash

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--repair-results',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    pp=sorted(a.results.glob('*.json'));rr=[json.loads(p.read_text()) for p in pp]
    if len(rr)!=10:raise ValueError('all ten controls required')
    ledger=[json.loads(x) for x in (a.results/'access_ledger.jsonl').read_text().splitlines()]
    for p,r in zip(pp,rr):
        if not any(e.get('case_id')==r['case_id'] and e.get('result_sha256')==file_hash(p) for e in ledger):raise ValueError('missing view receipt')
    stats={}
    for trust in [.75,.5,.25,.1]:
        rows=[]
        for r in rr:
            rp=json.loads((a.repair_results/(r['case_id']+'.json')).read_text())
            if file_hash(a.repair_results/(r['case_id']+'.json'))!=r['source_repair_result_sha256']:raise ValueError('source repair changed')
            ev=next(t['evaluation'] for t in r['trials'] if t['trust']==trust)
            result=evaluate(ev,rp['step0'],rp['offline_clean_reference']['actions'])
            rows.append({'case_id':r['case_id'],'factor':r['factor'],'baseline_success':rp['baseline_success'],'metrics':result,'gradient':r['heldout_relative_energy_derivative_at_identity']})
        failures=[r for r in rows if not r['baseline_success']]
        stats[str(trust)]={'mean_xyz_improvement_pct':float(np.mean([-r['metrics']['clipped']['xyz']['distance_change_pct'] for r in failures])),
            'n_material_improve':sum(r['metrics']['clipped']['xyz']['distance_change_pct']<=-10 for r in failures),
            'mean_relative_curvature_change_pct':float(np.mean([r['metrics']['relative_curvature']['change_pct'] for r in failures])),'rows':rows}
    a.out.write_text(json.dumps({'status':'post-grid mechanism control; no configuration selection','receipt_checks_pass':True,'n_dit_blocks':sorted(set(r['n_dit_blocks'] for r in rr)),'summary':stats},indent=2)+'\n')
