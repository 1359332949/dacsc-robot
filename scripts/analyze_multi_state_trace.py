"""Multi-state trace summary + closed-loop action/EEF divergence alignment."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def mean(x): return float(np.mean(x))

def load_trace(path):
    d=json.loads(path.read_text()); rs=d['records']
    out={}
    for branch,key in (('T','clean_vs_transform'),('S','clean_vs_shift')):
        out[branch]={
            'dino_rel':mean([mean([r['dino'][v][key] for v in r['dino']]) for r in rs]),
            'resampler_rel':mean([r['resampler'][key] for r in rs]),
            'action_xyz':mean([r['action'][key]['pre']['xyz'] for r in rs]),
            'action_rot':mean([r['action'][key]['pre']['rot'] for r in rs]),
            'action_grip_pre':mean([r['action'][key]['pre']['grip'] for r in rs]),
            'action_grip_post':mean([r['action'][key]['post']['grip'] for r in rs]),
        }
    out['target_contact_steps']=int(sum(bool(r['target_finger_contact']) for r in rs))
    out['n_steps']=len(rs); out['pairing_pass']=int(sum(bool(r['pairing_pass']) for r in rs))
    out['distance_start']=float(rs[0]['eef_target_distance']); out['distance_end']=float(rs[-1]['eef_target_distance'])
    return out

def arr(rec,k):
    if k=='action': return np.array([r['clipped_action'] for r in rec['trajectory']],float)
    return np.array([r[k] for r in rec['trajectory']],float)

def first(cond):
    idx=np.where(cond)[0]; return int(idx[0])+1 if len(idx) else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--trace-dir',type=Path,required=True); ap.add_argument('--failure-traj',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    trace={}
    for sid in (36,37,43,45): trace[str(sid)]=load_trace(args.trace_dir/f's{sid}.json')
    f=json.loads(args.failure_traj.read_text()); by={(r['state_id'],r['domain'],r['mode']):r for r in f['records']}
    align={}
    for state,dom in ((37,'clean'),(37,'shift'),(43,'shift'),(45,'shift')):
        a=by.get((state,dom,'frozen')); b=by.get((state,dom,'transform'))
        if not a or not b: continue
        af=arr(a,'action'); at=arr(b,'action'); n=min(len(af),len(at)); ad=np.max(np.abs(af[:n]-at[:n]),axis=1)
        ef=np.array([r['eef_pos'] for r in a['trajectory']],float); et=np.array([r['eef_pos'] for r in b['trajectory']],float); ed=np.linalg.norm(ef[:n]-et[:n],axis=1)
        align[f'{state}/{dom}']={'frozen_success':a['success'],'transform_success':b['success'],
            'first_eef_divergence_1mm':first(ed>0.001),
            'first_action_divergence_0.01':first(ad>0.01),
            'first_action_divergence_0.05':first(ad>0.05),
            'first_action_divergence_0.1':first(ad>0.1),
            'max_action_divergence':float(ad.max()),'max_eef_divergence':float(ed.max())}
    out={'schema':'multi_state_trace_summary_v1','trace':trace,'closed_loop_alignment':align}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2)[:6000])

if __name__=='__main__': main()
