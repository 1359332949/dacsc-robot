"""Retrospective primary-data checks; paired Clean actions are predictions, not GT."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.flow_geometry import auc

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--flow-results',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    m=json.loads(a.manifest.read_text());cases={c['case_id']:c for c in m['cases']};rr=[json.loads(p.read_text()) for p in a.flow_results.glob('*.json')]
    clean={cases[r['case_id']]['pair_group_id']:r for r in rr if r['factor']=='clean'};result={}
    for factor in ('sensor_noise','background','lighting'):
        rows=[r for r in rr if r['factor']==factor];cs=[];ratio=[];error={k:[] for k in ('xyz','rotation','all')};temporal=[];yy=[]
        for r in rows:
            cr=clean[cases[r['case_id']]['pair_group_id']]
            if r['z0_hashes']!=cr['z0_hashes']:raise ValueError('paired noise not identical')
            c=np.mean([v['all']['velocity_variance'] for v in r['metrics']]);cc=np.mean([v['all']['velocity_variance'] for v in cr['metrics']]);cs.append(float(c));ratio.append(float(c/cc));yy.append(not r['baseline_success'])
            actions=np.array(r['controller_actions']);ca=np.array(cr['controller_actions']);d=actions-ca
            for name,sl in [('xyz',slice(0,3)),('rotation',slice(3,6)),('all',slice(None))]:error[name].append(float(np.linalg.norm(d[:,:,sl].reshape(4,-1),axis=1).mean()))
            v=np.asarray(r['velocity_trace']);contrib=((v-v.mean(axis=1,keepdims=True))**2).mean(axis=(0,2,3,4));temporal.append(contrib)
        y=np.array(yy);temporal=np.stack(temporal)
        result[factor]={'n':len(rows),'m4_curvature_auc':auc(yy,cs),'paired_clean_normalized_m4_auc':auc(yy,ratio),
          'native_unclipped_reference_distance_aucs':{k:auc(yy,v) for k,v in error.items()},
          'curvature_vs_reference_distance_spearman':{k:float(spearmanr(cs,v).statistic) for k,v in error.items()},
          'normalized_curvature_vs_reference_distance_spearman':{k:float(spearmanr(ratio,v).statistic) for k,v in error.items()},
          'ratio_median_fail':float(np.median(np.array(ratio)[y])),'ratio_median_success':float(np.median(np.array(ratio)[~y])),
          'mean_fail_over_success_velocity_deviation_by_flow_step':(temporal[y].mean(0)/temporal[~y].mean(0)).tolist(),
          'warning':'mean step-energy ratios can be driven by outliers; not proof of a late-time causal repair mechanism'}
    a.out.write_text(json.dumps({'status':'retrospective exploratory audit; no deployment threshold fitted','distance_definition':'mean over four identical z0 of flattened full-chunk un-clipped controller L2',
        'reference_warning':'successful Clean policy output is a privileged prediction reference, not expert true action','factors':result},indent=2)+'\n')
