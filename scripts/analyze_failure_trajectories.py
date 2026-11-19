"""Analyze instrumentation-only failure trajectories."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def arrays(rec):
    tr=rec['trajectory']; n=len(tr)
    return {k:np.array([r[k] for r in tr],dtype=float) for k in ('raw_action','clipped_action','eef_pos','eef_quat','gripper_qpos','ncon','body_xpos','reward')}

def first(cond):
    idx=np.where(cond)[0]; return int(idx[0])+1 if len(idx) else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--traj',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    d=json.loads(args.traj.read_text()); recs=d['records']; by={(r['state_id'],r['domain'],r['mode']):r for r in recs}
    out={'schema':'failure_trajectory_summary_v1','cases':{},'paired':{}}
    for r in recs:
        a=arrays(r); tr=r['trajectory']; body=r['body_names']; n=len(tr)
        object_names=[i for i,nm in enumerate(body) if any(k in nm for k in ('bowl','ramekin','plate','cookie'))]
        z=a['body_xpos'][:,object_names,2] if object_names else np.zeros((n,0))
        out['cases'][f"{r['state_id']}/{r['domain']}/{r['mode']}"]={
            'success':r['success'],'steps':r['steps'],
            'first_close_command_step':first(a['raw_action'][:,6]>0.0),
            'first_contact_step':first(a['ncon']>0),
            'max_ncon':int(a['ncon'].max()),
            'eef_displacement':float(np.linalg.norm(a['eef_pos'][-1]-a['eef_pos'][0])),
            'max_eef_z':float(a['eef_pos'][:,2].max()),
            'gripper_qpos_min':float(a['gripper_qpos'].min()),
            'gripper_qpos_max':float(a['gripper_qpos'].max()),
            'object_z_initial':float(z[0].mean()) if z.size else None,
            'object_z_max':float(z[:,].max()) if z.size else None,
            'object_z_final':float(z[-1].mean()) if z.size else None,
            'transform_clip_mean':r.get('transform_clip_mean'),
        }
    # paired frozen vs transform
    for state in (36,37,43,45):
        for dom in ('clean','shift'):
            f=by.get((state,dom,'frozen')); t=by.get((state,dom,'transform'))
            if not f or not t: continue
            af=arrays(f); at=arrays(t); n=min(len(f['trajectory']),len(t['trajectory']))
            eefd=np.linalg.norm(af['eef_pos'][:n]-at['eef_pos'][:n],axis=1)
            actd=np.max(np.abs(af['clipped_action'][:n]-at['clipped_action'][:n]),axis=1)
            gripd=np.abs(af['raw_action'][:n,6]-at['raw_action'][:n,6])
            out['paired'][f'{state}/{dom}']={
                'frozen_success':f['success'],'transform_success':t['success'],
                'first_eef_divergence_1mm':first(eefd>0.001),
                'first_action_divergence_0.1':first(actd>0.1),
                'first_gripper_divergence_0.1':first(gripd>0.1),
                'max_eef_divergence':float(eefd.max()) if n else None,
                'max_action_divergence':float(actd.max()) if n else None,
                'frozen_close':first(af['raw_action'][:,6]>0),
                'transform_close':first(at['raw_action'][:,6]>0),
            }
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2)[:6000])

if __name__=='__main__': main()
