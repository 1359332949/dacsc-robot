"""Offline decision audit: wrist-late state confounding + source paired action spread."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

DIMS=('norm','xyz','rot','grip')

def pearson(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: return None
    return float(np.corrcoef(x,y)[0,1])

def rankdata(x):
    order=np.argsort(np.argsort(x)); return order.astype(float)

def partial_state_norm(x,y,cov):
    """Residualize x,y on state dummies + covariate, return correlation of residuals."""
    states=sorted(set(cov['state']))
    if len(states)<2: return None
    D=np.zeros((len(x),len(states)-1))
    st_arr=np.array([tuple(v) for v in cov['state']],dtype=object)
    for i,s in enumerate(states[:-1]):
        D[:,i]=np.array([tuple(v)==tuple(s) for v in st_arr]).astype(float)
    Z=np.column_stack([np.ones(len(x)),D,cov['clean_norm']])
    def resid(v):
        beta=np.linalg.lstsq(Z,v,rcond=None)[0]; return v-Z@beta
    rx=resid(x); ry=resid(y)
    return pearson(rx,ry)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--attention-report',type=Path,required=True); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); rep=json.loads(args.attention_report.read_text()); rows=rep['records']; npz=np.load(args.pairs,allow_pickle=False); man=json.loads(args.pairs.with_suffix('.manifest.json').read_text()); recs=man['records']
    # map feature_row -> clean controller action norm (first executed 12, clipped)
    clean_norm={}
    for r in recs:
        i=int(r['feature_row']); a=np.asarray(npz['clean_controller_actions'][i],dtype=float)[:12]
        clean_norm[i]=float(np.sqrt(np.mean(np.clip(a,-1,1)**2)))
    # ---------- wrist late state/chunk ----------
    wrist={}
    sel=[r for r in rows if r['window']=='late']
    groups={}
    for r in sel:
        key=(r['role'],int(r['state_id'])); groups.setdefault(key,[]).append(r)
    keys=sorted(groups)
    state_rows={k:groups[k] for k in keys}
    x_all=[]; y_all={d:[] for d in DIMS}; cov={'state':[],'clean_norm':[]}
    for k in keys:
        for r in state_rows[k]:
            x_all.append(float(r['attention_metrics']['eye_in_hand']['js_mean']))
            for d in DIMS: y_all[d].append(float(r['action_exec'][d]))
            cov['state'].append(k); cov['clean_norm'].append(clean_norm[int(r['feature_row'])])
    x_all=np.array(x_all)
    wrist['n_chunks']=len(x_all); wrist['n_states']=len(keys)
    wrist['within_state']={}
    for d in DIMS:
        y=np.array(y_all[d]);
        # state-centered
        xc=[]; yc=[]
        for k in keys:
            idx=[i for i,(rk) in enumerate([(r['role'],int(r['state_id'])) for r in sel]) if rk==k]
            if len(idx)<2: continue
            xc.extend(x_all[idx]-x_all[idx].mean()); yc.extend(y[idx]-y[idx].mean())
        xc=np.array(xc); yc=np.array(yc)
        wrist['within_state'][d]={'pearson':pearson(xc,yc), 'spearman':float(spearmanr(xc,yc).statistic) if len(xc)>2 else None,
                                  'n_chunks':len(xc), 'partial_state_plus_clean_norm':partial_state_norm(x_all,y,cov)}
    # state-level raw (for comparison)
    wrist['state_level']={}
    for d in DIMS:
        ys=[]; xs=[]
        for k in keys:
            idx=[i for i,(rk) in enumerate([(r['role'],int(r['state_id'])) for r in sel]) if rk==k]
            xs.append(x_all[idx].mean()); ys.append(np.array(y_all[d])[idx].mean())
        wrist['state_level'][d]={'pearson':pearson(np.array(xs),np.array(ys))}
    # ---------- source paired action spread ----------
    spread={}
    for window in ('early','late'):
        recs_w=[r for r in recs if r.get('role')=='source_fit' and r.get('window')==window and r.get('pairing_pass')]
        vals={d:[] for d in DIMS}; bystate={}
        for r in recs_w:
            i=int(r['feature_row']); ac=np.clip(np.asarray(npz['clean_controller_actions'][i],dtype=float)[:12],-1,1); as_=np.clip(np.asarray(npz['shift_controller_actions'][i],dtype=float)[:12],-1,1); d=ac-as_
            v={'norm':float(np.sqrt(np.mean(d**2))),'xyz':float(np.sqrt(np.mean(d[:,:3]**2))),'rot':float(np.sqrt(np.mean(d[:,3:6]**2))),'grip':float(np.sqrt(np.mean(d[:,6]**2)))}
            for k in DIMS: vals[k].append(v[k])
            bystate.setdefault(int(r['state_id']),[]).append(v)
        out={'n_records':len(recs_w),'n_states':len(bystate)}
        for k in DIMS:
            arr=np.array(vals[k]); means=[]
            within=[]
            for s,vs in bystate.items():
                a=np.array([v[k] for v in vs]); means.append(a.mean()); within.extend(a-a.mean())
            out[k]={'mean':float(arr.mean()),'std':float(arr.std()),'min':float(arr.min()),'p50':float(np.median(arr)),'max':float(arr.max()),
                    'between_state_var':float(np.var(means)) if len(means)>1 else None,
                    'within_state_var':float(np.var(within)) if within else None,
                    'icc':float(np.var(means)/(np.var(means)+np.var(within)+1e-12)) if len(means)>1 and within else None}
        spread[window]=out
    result={'schema':'offline_decision_audit_v1','wrist_late_confounding':wrist,'source_paired_action_spread':spread}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)[:6000])

if __name__=='__main__': main()
