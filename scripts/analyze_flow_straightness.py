"""Prespecified descriptive t=0 flow geometry analysis; no model/threshold fitting."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.flow_geometry import auc
from dacsc.static_visual_v2 import file_hash


def paired_concordance(rows,scores,fields=('base_task',)):
    vals=[]
    for i,r in enumerate(rows):
        if r['baseline_success']:continue
        for j,s in enumerate(rows):
            if s['baseline_success'] and all(r[k]==s[k] for k in fields):
                vals.append(float(scores[i]>scores[j])+.5*float(scores[i]==scores[j]))
    return {'eligible_pairs':len(vals),'concordance':float(np.mean(vals)) if vals else None}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    paths=sorted(a.results.glob('*.json'));rows=[json.loads(p.read_text()) for p in paths]
    m=json.loads(a.manifest.read_text());expected={c['case_id'] for c in m['cases']}
    if len(rows)!=100 or {r['case_id'] for r in rows}!=expected or any(r['status']!='completed' for r in rows):raise ValueError('requires all 100 completed planned cases')
    if any(r['manifest_sha256']!=file_hash(a.manifest) for r in rows):raise ValueError('manifest mismatch')
    scores={}
    for channel in ('all','xyz','rotation','gripper'):
        for metric in ('velocity_variance','relative_variance','turning','path_excess'):
            for n in (1,4):
                name=f'{channel}.{metric}.m{n}'
                vv=[[r['metrics'][k][channel][metric] for k in range(n)] for r in rows]
                if any(any(v is None for v in vs) for vs in vv):continue
                scores[name]=np.array([np.mean(vs) for vs in vv])
    for name in ('endpoint_variance_xyz','endpoint_variance_rotation'):scores[name]=np.array([r[name] for r in rows])
    clean=np.array([r['factor']=='clean' for r in rows]);shift=~clean;report={}
    tasks=sorted({r['base_task'] for r in rows});rng=np.random.default_rng(20261124)
    boot=[rng.choice(tasks,len(tasks),replace=True).tolist() for _ in range(1000)]
    for name,sc in scores.items():
        threshold=float(sc[clean].max());d={'clean_median':float(np.median(sc[clean])),'shift_median':float(np.median(sc[shift])),
        'domain_shift_auc':auc(shift,sc),'threshold_strictly_above_clean_max':threshold,'clean_resubstitution_fpr':float(np.mean(sc[clean]>threshold)),'factors':{}}
        for factor in ('sensor_noise','background','lighting'):
            ids=[i for i,r in enumerate(rows) if r['factor']==factor];rr=[rows[i] for i in ids];ss=sc[ids];yy=np.array([not r['baseline_success'] for r in rr]);co=paired_concordance(rr,ss)
            vals=[]
            for bt in boot:
                ix=[j for t in bt for j,r in enumerate(rr) if r['base_task']==t]
                v=auc(yy[ix],ss[ix])
                if v is not None:vals.append(v)
            d['factors'][factor]={'n':len(ids),'failures':int(yy.sum()),'failure_auc':auc(yy,ss),'same_task':co,
                'same_task_and_subtype':paired_concordance(rr,ss,('base_task','sub_dimension')),
                'same_task_and_noise_code':paired_concordance(rr,ss,('base_task','noise_code')) if factor=='sensor_noise' else None,
                'failure_recall':float(np.mean(ss[yy]>threshold)),'shift_success_fpr':float(np.mean(ss[~yy]>threshold)),
                'cluster_bootstrap_auc_95':np.quantile(vals,[.025,.975]).tolist() if vals else None,'bootstrap_valid':len(vals),
                'noise_code_coverage':dict(Counter(str(r['noise_code']) for r in rr)),
                'subtypes':{sub:{'n':sum(r['sub_dimension']==sub for r in rr),
                    'failures':sum(not r['baseline_success'] for r in rr if r['sub_dimension']==sub),
                    'auc':auc([not r['baseline_success'] for r in rr if r['sub_dimension']==sub],ss[[j for j,r in enumerate(rr) if r['sub_dimension']==sub]])} for sub in sorted({r['sub_dimension'] for r in rr})}}
        report[name]=d
    primary=report['all.velocity_variance.m1'];relative=report['all.relative_variance.m1'];passed=[]
    for f,d in primary['factors'].items():
        if d['failure_auc']>=.7 and d['same_task']['eligible_pairs']>=5 and d['same_task']['concordance']>=.6 and d['failure_recall']>=.5 and relative['factors'][f]['failure_auc']>.5:passed.append(f)
    out={'status':'EXPLORATORY_GATE_PASS' if passed else 'STOP_FLOW_ENERGY_TTA_V1','passed_factors':passed,
         'primary_score':'all.velocity_variance.m1','rows':len(rows),'factor_counts':dict(Counter(r['factor'] for r in rows)),
         'max_predict_parity_error':max(r['official_controller_max_abs'] for r in rows),
         'scores':report,'inputs':{p.name:file_hash(p) for p in paths},
         'limitations':['Dev reuse; multiple prespecified scores are descriptive','bootstrap has only 10 base-task clusters; no population guarantee',
                         'threshold estimated and evaluated on same 10 Clean cases','t=0 final outcome association; no causal risk event/lead time',
                         'gradient localization requires external spatial ground truth; not inferred from a heatmap']}
    a.out.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('scores','inputs')}));print(json.dumps(primary,indent=2))
if __name__=='__main__':main()
