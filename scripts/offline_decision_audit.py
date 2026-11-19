"""Offline decision audit: wrist-late state confounding + source action spread.

All grouping uses explicit aligned indices; input record order must not affect
the statistics.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

DIMS=('norm','xyz','rot','grip')
ROLES=('source_fit','source_validation')


def pearson(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: return None
    return float(np.corrcoef(x,y)[0,1])


def partial_state_norm(x,y,states,clean_norm):
    """Residualize x,y on state dummies + clean action norm; correlate residuals."""
    keys=sorted(set(tuple(x) for x in states))
    if len(keys)<2: return None
    D=np.zeros((len(x),len(keys)-1))
    for i,k in enumerate(keys[:-1]): D[:,i]=np.array([tuple(x)==k for x in states],dtype=float)
    Z=np.column_stack([np.ones(len(x)),D,clean_norm])
    def resid(v):
        beta=np.linalg.lstsq(Z,v,rcond=None)[0]; return v-Z@beta
    return pearson(resid(x),resid(y))


def compute_wrist_late(rows):
    """Pure chunk-level function; returns dict. Rows need pairing_pass, role, window."""
    selected=[r for r in rows if r['window']=='late']
    for r in selected:
        if not bool(r.get('pairing_pass', False)):
            raise RuntimeError('failed pair in wrist-late audit')
        if r.get('role') not in ROLES:
            raise RuntimeError(f'unexpected role {r.get("role")}')
    groups={}
    for r in selected:
        groups.setdefault((r['role'],int(r['state_id'])),[]).append(r)
    keys=sorted(groups)
    aligned=[]; group_indices={k:[] for k in keys}
    for k in keys:
        for r in groups[k]:
            group_indices[k].append(len(aligned)); aligned.append(r)
    x=np.array([float(r['attention_metrics']['eye_in_hand']['js_mean']) for r in aligned])
    ys={d:np.array([float(r['action_exec'][d]) for r in aligned]) for d in DIMS}
    clean=np.array([float(r['_clean_action_norm']) for r in aligned])
    states=[(r['role'],int(r['state_id'])) for r in aligned]
    out={'n_chunks':len(aligned),'n_states':len(keys),'within_state':{},'state_level':{}}
    for d in DIMS:
        y=ys[d]; xc=[]; yc=[]
        for k in keys:
            idx=group_indices[k]
            if len(idx)<2: continue
            xc.extend(x[idx]-x[idx].mean()); yc.extend(y[idx]-y[idx].mean())
        xc=np.array(xc); yc=np.array(yc)
        out['within_state'][d]={
            'pearson':pearson(xc,yc),
            'spearman':float(spearmanr(xc,yc).statistic) if len(xc)>2 and np.std(xc)>0 and np.std(yc)>0 else None,
            'n_chunks':int(len(xc)),
            'partial_state_plus_clean_norm':partial_state_norm(x,ys[d],states,clean),
        }
        xs=[]; yv=[]
        for k in keys:
            idx=group_indices[k]; xs.append(float(x[idx].mean())); yv.append(float(ys[d][idx].mean()))
        out['state_level'][d]={'pearson':pearson(np.array(xs),np.array(yv))}
    return out


def compute_spread(records, npz):
    out={}
    for window in ('early','late'):
        rs=[r for r in records if r.get('role')=='source_fit' and r.get('window')==window]
        if not all(r.get('pairing_pass',False) for r in rs):
            raise RuntimeError('failed pair in spread audit')
        vals={d:[] for d in DIMS}; bystate={}
        for r in rs:
            i=int(r['feature_row'])
            ac=np.clip(np.asarray(npz['clean_controller_actions'][i],dtype=float)[:12],-1,1)
            ash=np.clip(np.asarray(npz['shift_controller_actions'][i],dtype=float)[:12],-1,1)
            d=ac-ash
            v={'norm':float(np.sqrt(np.mean(d**2))),'xyz':float(np.sqrt(np.mean(d[:,:3]**2))),
               'rot':float(np.sqrt(np.mean(d[:,3:6]**2))),'grip':float(np.sqrt(np.mean(d[:,6]**2)))}
            for k in DIMS: vals[k].append(v[k])
            bystate.setdefault(int(r['state_id']),[]).append(v)
        res={'n_records':len(rs),'n_states':len(bystate),
             'action_input_semantics':'controller_actions clipped to [-1,1], first 12 steps; stored pre-env controller command'}
        for k in DIMS:
            a=np.array(vals[k]); means=[]; within=[]
            for s,vs in bystate.items():
                av=np.array([v[k] for v in vs]); means.append(av.mean()); within.extend(av-av.mean())
            means=np.array(means); within=np.array(within)
            var_between=float(means.var()) if len(means)>1 else None
            var_within=float(within.var()) if len(within) else None
            res[k]={'mean':float(a.mean()),'std':float(a.std()),'min':float(a.min()),
                    'p50':float(np.median(a)),'p75':float(np.quantile(a,.75)),'p90':float(np.quantile(a,.90)),
                    'p95':float(np.quantile(a,.95)),'max':float(a.max()),
                    'prob_gt_0':float(np.mean(a>0)),'prob_gt_0.1':float(np.mean(a>0.1)),
                    'prob_gt_0.5':float(np.mean(a>0.5)),
                    'between_state_variance_share':(var_between/(var_between+var_within+1e-12)) if var_between is not None and var_within is not None else None}
        out[window]=res
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--attention-report',type=Path,required=True)
    ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); rep=json.loads(args.attention_report.read_text()); rows=rep['records']
    npz=np.load(args.pairs,allow_pickle=False); man=json.loads(args.pairs.with_suffix('.manifest.json').read_text()); recs=man['records']
    # Attach clean action norm by unique feature row and validate alignment.
    by_row={int(r['feature_row']):r for r in recs}
    if len(by_row)!=len(recs): raise RuntimeError('feature_row not unique in manifest')
    # Attention report rows must map to manifest and be passed pairs.
    for r in rows:
        if int(r['feature_row']) not in by_row: raise RuntimeError('attention row missing from manifest')
        if not bool(r.get('pairing_pass',False)): raise RuntimeError('attention report contains failed pair')
    for r in rows:
        r['_clean_action_norm']=float(np.sqrt(np.mean(np.clip(np.asarray(npz['clean_controller_actions'][int(r['feature_row'])],dtype=float)[:12],-1,1)**2)))
    result={'schema':'offline_decision_audit_v1',
            'wrist_late_confounding':compute_wrist_late(rows),
            'source_paired_action_spread':compute_spread(recs,npz)}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2)[:6000])

if __name__=='__main__': main()
