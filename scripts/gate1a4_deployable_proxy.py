"""Gate 1A-4: deployable (no Clean at test time) attention priors/proxies."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

METRIC_NAMES=('entropy_mean','maxprob_mean','lse_mean')

def vec(d): return np.array([d[k] for k in METRIC_NAMES],dtype=float)

def pearson(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: return None
    return float(np.corrcoef(x,y)[0,1])

def boot_ci(x,y,rng,n=2000):
    vals=[]
    for _ in range(n):
        i=rng.integers(0,len(x),len(x))
        r=pearson(x[i],y[i])
        if r is not None: vals.append(r)
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))] if vals else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--report',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); d=json.loads(args.report.read_text()); rows=d['records']; views=set()
    for r in rows:
        if not bool(r.get('pairing_pass', False)):
            raise RuntimeError(f"Gate 1A-4 received failed pair {r.get('role')}/{r.get('state_id')}/{r.get('chunk_index')}")
        if r.get('role') not in ('source_fit', 'source_validation'):
            raise RuntimeError(f"unexpected role {r.get('role')}")
        views.update(r['attention_metrics'].keys())
    # Clean/Shift rows are paired by construction; still check state/chunk coverage.
    for key in (('source_fit',0),('source_validation',0)):
        pass
    out={'schema':'gate1a4_deployable_proxy_v1','source_report':str(args.report),'views':{},'correlation':{}}
    for view in sorted(views):
        out['views'][view]={}
        for win in ('early','late'):
            # state-level descriptors: clean prior from source_fit; shift proxy
            cfit={}; cval={}; sfit={}; sval={}
            for r in rows:
                if r['window']!=win: continue
                key=(r['role'],int(r['state_id'])); c=r['attention_metrics'][view]['attention_clean']; s=r['attention_metrics'][view]['attention_shift']
                (cfit if r['role']=='source_fit' else cval).setdefault(key,[]).append(vec(c))
                (sfit if r['role']=='source_fit' else sval).setdefault(key,[]).append(vec(s))
            def avg(g): return {k:np.mean(v,axis=0) for k,v in g.items()}
            cfit=avg(cfit); cval=avg(cval); sfit=avg(sfit); sval=avg(sval)
            fit_ids=sorted(k[1] for k in cfit); val_ids=sorted(k[1] for k in cval)
            if not fit_ids or not val_ids: out['views'][view][win]={'status':'insufficient'}; continue
            prior=np.mean([cfit[( 'source_fit',i)] for i in fit_ids],axis=0)
            scale=np.maximum(np.std([cfit[('source_fit',i)] for i in fit_ids],axis=0),1e-4)
            def dist(v): return float(np.mean(((v-prior)/scale)**2))
            fcs={i:dist(cfit[('source_fit',i)]) for i in fit_ids}
            vcs={i:dist(cval[('source_validation',i)]) for i in val_ids}
            fss={int(k[1]):dist(v) for k,v in sfit.items()}
            vss={int(k[1]):dist(v) for k,v in sval.items()}
            tau=float(max(fcs.values()))
            gate={'tau':tau,'fit_clean_fpr':float(np.mean(np.array(list(fcs.values()))>tau)),
                  'val_clean_fpr':float(np.mean(np.array(list(vcs.values()))>tau)),
                  'fit_shift_activation':float(np.mean(np.array(list(fss.values()))>tau)),
                  'val_shift_activation':float(np.mean(np.array(list(vss.values()))>tau))}
            try:
                from sklearn.metrics import roc_auc_score
                gate['val_auroc']=float(roc_auc_score([0]*len(vcs)+[1]*len(vss),list(vcs.values())+list(vss.values())))
            except Exception: gate['val_auroc']=None
            out['views'][view][win]=gate
    # correlation: shift-prior distance and action deviation, state-level
    for view in sorted(views):
        for win in ('early','late'):
            groups={}
            for r in rows:
                if r['window']!=win: continue
                s=r['attention_metrics'][view]['attention_shift']; key=(r['role'],int(r['state_id']))
                groups.setdefault(key,[]).append((vec(s),r['action_exec']))
            # build prior from source_fit clean
            cfit={}
            for r in rows:
                if r['window']!=win or r['role']!='source_fit': continue
                cfit.setdefault(int(r['state_id']),[]).append(vec(r['attention_metrics'][view]['attention_clean']))
            fit_ids=sorted(cfit)
            if len(fit_ids)<2: continue
            prior=np.mean([np.mean(cfit[i],axis=0) for i in fit_ids],axis=0)
            scale=np.maximum(np.std([np.mean(cfit[i],axis=0) for i in fit_ids],axis=0),1e-4)
            keys=sorted(groups)
            xs=[]; ys={k:[] for k in ('norm','xyz','rot','grip')}
            for key in keys:
                vals=groups[key]
                sv=np.mean([v[0] for v in vals],axis=0)
                xs.append(float(np.mean(((sv-prior)/scale)**2)))
                for dim in ys: ys[dim].append(float(np.mean([v[1][dim] for v in vals])))
            xs=np.array(xs)
            for dim,yv in ys.items():
                yv=np.array(yv); rho=pearson(xs,yv); sp=spearmanr(xs,yv).statistic if rho is not None else None
                fit_idx=[i for i,k in enumerate(keys) if k[0]=='source_fit']; val_idx=[i for i,k in enumerate(keys) if k[0]=='source_validation']
                out['correlation'][f'{view}/{win}/shift_prior/{dim}']={'n_states':len(keys),'pearson':rho,
                    'spearman':float(sp) if sp is not None else None,
                    'pearson_ci95':boot_ci(xs,yv,np.random.default_rng(20261119)) if rho is not None else None,
                    'fit_pearson':pearson(xs[fit_idx],yv[fit_idx]),'val_pearson':pearson(xs[val_idx],yv[val_idx])}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'views':out['views'],'correlation':out['correlation']},indent=2)[:5000])

if __name__=='__main__': main()
