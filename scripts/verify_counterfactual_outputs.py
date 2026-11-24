"""CPU-only re-read verification against saved inputs and frozen output arrays."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dacsc.counterfactual_diagnostic import command_difference,rms,require_current_pair,interaction
from dacsc.interaction_records import array_hash

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--outputs',type=Path,required=True);ap.add_argument('--paired-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    r=json.loads(a.report.read_text());assert hashlib.sha256(a.outputs.read_bytes()).hexdigest()==r['outputs_sha256']
    with np.load(a.outputs,allow_pickle=False) as z:outputs={k:z[k] for k in z.files}
    assert array_hash(outputs['x_probe'])==r['x_probe_sha256'];assert array_hash(outputs['z0'])==r['z0_sha256']
    errors=[];pairs={}
    for sid,relative in r['config']['canonical_sources'].items():
        pairs[int(sid)]={}
        for dom in ('clean','shift'):
            p=a.paired_root/relative/(dom+'.npz')
            assert hashlib.sha256(p.read_bytes()).hexdigest()==r['input_metadata'][sid][dom]['npz_sha256']
            with np.load(p,allow_pickle=False) as z:pairs[int(sid)][dom]={k:z[k] for k in z.files}
    for row in r['records']:
        sid,t,i=row['state_id'],row['step'],row['boundary_index'];pair=pairs[sid]
        q=require_current_pair(pair['clean'],pair['shift'],i);np.testing.assert_array_equal(q,row['current_state8'])
        for v in ('CC','SC','CS','SS'):
            assert row['checks'][v+'_state_input_hash']==array_hash(q)
            if v in ('CC','SS'):
                dom='clean' if v=='CC' else 'shift';saved=np.concatenate([pair[dom]['tokens_'+view][i] for view in r['config']['views']])
                np.testing.assert_allclose(outputs[f's{sid}_t{t}_{v}_tokens'],saved,atol=1e-5,rtol=0)
        for v in ('SC','CS','SS'):
            result=command_difference(outputs[f's{sid}_t{t}_{v}_actions'],outputs[f's{sid}_t{t}_CC_actions'])
            for clip in ('preclip','postclip'):
                for k in ('all_rms','xyz_l2_mean','rotation_channels_l2_mean','gripper_abs_mean'):
                    errors.append(abs(result[clip][k]-row['variants'][v]['command'][clip][k]))
            vel=rms(outputs[f's{sid}_t{t}_{v}_velocity']-outputs[f's{sid}_t{t}_CC_velocity'])
            errors.append(abs(vel-row['variants'][v]['same_x_velocity_rms']))
    assert max(errors)<1e-12
    assert r['parameters_before']==r['parameters_after']
    out={'pass':True,'scope':'CPU re-read current-proprio/input/output pairing and published command/velocity metrics',
         'metrics_compared':len(errors),'max_metric_abs_diff':max(errors),'unique_boundaries':len(r['records']),
         'independent_initial_states':2,'outputs_sha256':r['outputs_sha256'],'parameters_unchanged':True}
    a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
