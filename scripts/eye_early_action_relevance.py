"""Eye-in-hand early domain score vs executed action deviation (validation only)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def boot(x,y,rng,n=4000):
    vals=[]
    for _ in range(n):
        i=rng.integers(0,len(x),len(x))
        if np.std(x[i])==0 or np.std(y[i])==0: continue
        vals.append(float(np.corrcoef(x[i],y[i])[0,1]))
    return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))] if vals else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gate1a',type=Path,required=True); ap.add_argument('--action',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); g=json.loads(args.gate1a.read_text()); a=json.loads(args.action.read_text())
    ps=g['views']['eye_in_hand']['early']['per_state']
    # keys are string state ids for val_shift
    val_scores={int(k):float(v) for k,v in ps['val_shift'].items()}
    clean_scores={int(k):float(v) for k,v in ps['val_clean'].items()}
    dims=('norm','xyz','rot','grip')
    acts={d:{} for d in dims}
    for r in a['records']:
        if r['role']!='source_validation' or r['window']!='early': continue
        sid=int(r['state_id']); acts['norm'].setdefault(sid,[]).append(r['action_exec']['norm'])
        for d in dims[1:]: acts[d].setdefault(sid,[]).append(r['action_exec'][d])
    for d in dims: acts[d]={k:float(np.mean(v)) for k,v in acts[d].items()}
    ids=sorted(set(val_scores)&set(acts['norm']))
    x=np.array([val_scores[i] for i in ids]); out={'schema':'eye_early_action_relevance_v1','n_states':len(ids),'states':ids,
        'eye_early_shift_score':x.tolist(),'eye_early_clean_score':[clean_scores.get(i) for i in ids],
        'action_exec':{d:[acts[d][i] for i in ids] for d in dims},'correlation':{}}
    rng=np.random.default_rng(20261119)
    for d in dims:
        y=np.array([acts[d][i] for i in ids])
        rho=float(np.corrcoef(x,y)[0,1]) if len(x)>2 and x.std()>0 and y.std()>0 else None
        sp=float(spearmanr(x,y).statistic) if rho is not None else None
        out['correlation'][d]={'pearson':rho,'spearman':sp,'pearson_ci95':boot(x,y,rng) if rho is not None else None}
    # high/low score groups
    med=float(np.median(x)); hi=[i for i,v in enumerate(x) if v>=med]; lo=[i for i,v in enumerate(x) if v<med]
    out['high_low']={d:{'high_mean':float(np.mean([acts[d][ids[i]] for i in hi])), 'low_mean':float(np.mean([acts[d][ids[i]] for i in lo])), 'high_n':len(hi),'low_n':len(lo)} for d in dims}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'n':len(ids),'states':ids,'corr':out['correlation'],'high_low':out['high_low']},indent=2))

if __name__=='__main__': main()
