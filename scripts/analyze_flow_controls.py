"""Post-hoc explanatory controls, explicitly excluded from primary selection."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.flow_geometry import auc

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    rr=[json.loads(p.read_text()) for p in sorted(a.results.glob('*.json'))];out={}
    if len(rr)!=100 or any(r['status']!='completed' for r in rr):raise ValueError('100 completed cases required')
    for f in ['sensor_noise','background','lighting']:
        r=[x for x in rr if x['factor']==f];y=[not x['baseline_success'] for x in r]
        curves=[x['metrics'][0]['all']['velocity_variance'] for x in r]
        lowrms=[-float(np.sqrt(np.mean(np.clip(np.array(x['controller_actions'])[0,:,:3],-1,1)**2))) for x in r]
        v0=[float(np.mean(np.array(x['velocity_trace'])[0,0]**2)) for x in r]
        v9=[float(np.mean(np.array(x['velocity_trace'])[0,-1]**2)) for x in r]
        out[f]={'primary_curvature_auc':auc(y,curves),'posthoc_lower_xyz_rms_auc':auc(y,lowrms),
                'posthoc_initial_velocity_energy_auc':auc(y,v0),'posthoc_final_velocity_energy_auc':auc(y,v9)}
    a.out.write_text(json.dumps({'status':'posthoc explanatory controls; not used to choose primary score or gate','factors':out},indent=2)+'\n')
