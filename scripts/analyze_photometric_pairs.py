"""P2 analysis: state-balanced Clean/Shift image-stat differences."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

FEATURES=['lum_mean','lum_median','lum_std','lum_p10','lum_p90','contrast_p90_p10',
          'saturation_mean','shadow_fraction','highlight_fraction','mean_gradient_magnitude','edge_std',
          'local_lum_std_mean','local_lum_std_max']
# Legacy raw-file key for gradient magnitude.
RAW_KEY={'mean_gradient_magnitude':'edge_mean'}
VEC_FEATURES=['rgb_mean','rgb_std']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fit',type=Path,required=True); ap.add_argument('--val',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); rows=[]
    for p in (args.fit,args.val):
        d=json.loads(p.read_text()); rows+=d['records']
    for r in rows:
        if not r.get('pairing_pass',False): raise RuntimeError('failed photometric pair')
        if r.get('role') not in ('source_fit','source_validation'): raise RuntimeError('bad role')
        for key in ('view','window','chunk','state_id'):
            if key not in r: raise RuntimeError(f'missing {key} in photometric record')
    # Fit/validation files must be role-pure and disjoint in state ids.
    fit_rows=[r for r in json.loads(args.fit.read_text())['records']]
    val_rows=[r for r in json.loads(args.val.read_text())['records']]
    if any(r['role']!='source_fit' for r in fit_rows): raise RuntimeError('fit file contains non-source_fit rows')
    if any(r['role']!='source_validation' for r in val_rows): raise RuntimeError('val file contains non-source_validation rows')
    if {int(r['state_id']) for r in fit_rows} & {int(r['state_id']) for r in val_rows}: raise RuntimeError('fit/val state ids overlap')
    unique=set()
    for r in rows:
        key=(r['role'],int(r['state_id']),int(r['chunk']),r['view'])
        if key in unique: raise RuntimeError(f'duplicate photometric record {key}')
        unique.add(key)
    out={'schema':'photometric_analysis_v1','views':{}}
    for view in ('agentview','eye_in_hand'):
        out['views'][view]={}
        for window in ('early','late'):
            sel=[r for r in rows if r['view']==view and r['window']==window]
            bystate={}
            for r in sel: bystate.setdefault((r['role'],int(r['state_id'])),[]).append(r)
            res={'n_records':len(sel),'n_states':len(bystate),'features':{}}
            for feat in FEATURES:
                raw=RAW_KEY.get(feat,feat)
                per_role={}
                for role in ('source_fit','source_validation'):
                    states=[]
                    for (rr,sid),rs in bystate.items():
                        if rr!=role: continue
                        d=[(r['shift'][raw]-r['clean'][raw]) for r in rs]
                        states.append((sid,float(np.mean(d))))
                    if not states: continue
                    vals=np.array([v for _,v in states])
                    rng=np.random.default_rng(20261119); boots=[float(np.mean(rng.choice(vals,len(vals),replace=True))) for _ in range(4000)]
                    clean=np.array([np.mean([r['clean'][raw] for r in bystate[(role,s)]]) for s,_ in states]); shift=np.array([np.mean([r['shift'][raw] for r in bystate[(role,s)]]) for s,_ in states])
                    per_role[role]={'mean_delta':float(vals.mean()),'std_delta':float(vals.std()),'median_delta':float(np.median(vals)),
                                    'mean_delta_ci95':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],
                                    'fraction_positive':float((vals>0).mean()),'mean_clean':float(clean.mean()),'mean_shift':float(shift.mean()),
                                    'cohens_dz_descriptive':float(vals.mean()/(vals.std()+1e-9)),'n_states':len(states)}
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
            for feat in ['lum_mean','lum_std','contrast_p90_p10','saturation_mean','shadow_fraction','mean_gradient_magnitude','local_lum_std_max']:
                y=x['features'][feat]
                for role in ('source_fit','source_validation'):
                    if role in y:
                        q=y[role]; print(feat,role,'delta',round(q['mean_delta'],4),'ci', [round(v,4) for v in q['mean_delta_ci95']], 'frac+',round(q['fraction_positive'],2))
            print('rgb',x['rgb'])

if __name__=='__main__': main()
