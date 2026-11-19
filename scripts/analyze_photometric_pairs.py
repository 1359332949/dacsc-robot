"""P2 analysis: state-balanced Clean/Shift image-stat differences."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

FEATURES=['lum_mean','lum_median','lum_std','lum_p10','lum_p90','contrast_p90_p10',
          'saturation_mean','shadow_fraction','highlight_fraction','edge_mean','edge_std',
          'local_lum_std_mean','local_lum_std_max','local_edge_mean_mean']
VEC_FEATURES=['rgb_mean','rgb_std']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fit',type=Path,required=True); ap.add_argument('--val',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); rows=[]
    for p in (args.fit,args.val):
        d=json.loads(p.read_text()); rows+=d['records']
    for r in rows:
        if not r.get('pairing_pass',False): raise RuntimeError('failed photometric pair')
        if r.get('role') not in ('source_fit','source_validation'): raise RuntimeError('bad role')
    out={'schema':'photometric_analysis_v1','views':{}}
    for view in ('agentview','eye_in_hand'):
        out['views'][view]={}
        for window in ('early','late'):
            sel=[r for r in rows if r['view']==view and r['window']==window]
            bystate={}
            for r in sel: bystate.setdefault((r['role'],int(r['state_id'])),[]).append(r)
            res={'n_records':len(sel),'n_states':len(bystate),'features':{}}
            for feat in FEATURES:
                per_role={}
                for role in ('source_fit','source_validation'):
                    states=[]
                    for (rr,sid),rs in bystate.items():
                        if rr!=role: continue
                        d=[(r['shift'][feat]-r['clean'][feat]) for r in rs]
                        states.append((sid,float(np.mean(d))))
                    if not states: continue
                    vals=np.array([v for _,v in states]); clean=np.array([np.mean([r['clean'][feat] for r in bystate[(role,s)]]) for s,_ in states]); shift=np.array([np.mean([r['shift'][feat] for r in bystate[(role,s)]]) for s,_ in states])
                    per_role[role]={'mean_delta':float(vals.mean()),'std_delta':float(vals.std()),'median_delta':float(np.median(vals)),
                                    'fraction_positive':float((vals>0).mean()),'mean_clean':float(clean.mean()),'mean_shift':float(shift.mean()),
                                    'cohens_dz':float(vals.mean()/(vals.std()+1e-9)),'n_states':len(states)}
                res['features'][feat]=per_role
            # RGB vector differences (norm and per channel)
            recs_vectors={}
            for role in ('source_fit','source_validation'):
                ds=[]
                for (rr,sid),rs in bystate.items():
                    if rr!=role: continue
                    c=np.array([r['clean']['rgb_mean'] for r in rs]); s=np.array([r['shift']['rgb_mean'] for r in rs])
                    ds.append(np.linalg.norm(np.mean(s,axis=0)-np.mean(c,axis=0)))
                if ds: recs_vectors[role]={'rgb_shift_norm_mean':float(np.mean(ds)),'n_states':len(ds)}
            res['rgb']=recs_vectors
            out['views'][view][window]=res
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2)+'\n')
    for view,v in out['views'].items():
        for win,x in v.items():
            print('==',view,win,'states',x['n_states'])
            for feat in ['lum_mean','lum_std','contrast_p90_p10','saturation_mean','shadow_fraction','edge_mean','local_lum_std_max']:
                y=x['features'][feat]
                for role in ('source_fit','source_validation'):
                    if role in y:
                        q=y[role]; print(feat,role,'delta',round(q['mean_delta'],4),'frac+',round(q['fraction_positive'],2),'dz',round(q['cohens_dz'],2))
            print('rgb',x['rgb'])

if __name__=='__main__': main()
