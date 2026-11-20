"""Ridge R0 with Clean no-op, state-grouped alpha selection."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--use-visual',action='store_true'); ap.add_argument('--clean-weight',type=float,default=1.0); args=ap.parse_args()
    d=np.load(args.data); X=d['features'].astype(np.float64); Y=d['labels'].astype(np.float64); roles=d['roles']; states=d['states']; clean=d['clean'].astype(bool)
    if not args.use_visual: X=X[:,768:]  # h(768)+state8+tau+k
    tr=(roles=='source_fit'); va=(roles=='source_validation'); tr_shift=tr&~clean; va_shift=va&~clean
    mu=X[tr].mean(axis=0); sd=X[tr].std(axis=0); sd[sd<1e-8]=1; Xn=(X-mu)/sd
    def fit_ridge(idx,alpha):
        D=np.c_[Xn[idx],np.ones(idx.sum())]; w=np.where(clean[idx],args.clean_weight,1.0)
        A=D.T@(w[:,None]*D)+alpha*np.eye(D.shape[1]); b=D.T@(w[:,None]*Y[idx])
        return np.linalg.solve(A,b)
    # state-grouped CV on source_fit
    uniq=sorted(set(states[tr])); best=None
    for alpha in (1,10,100,1000,10000):
        errs=[]
        for s in uniq:
            hold=tr&(states==s); train=tr&(states!=s)
            beta=fit_ridge(train,alpha); p=(np.c_[Xn[hold],np.ones(hold.sum())]@beta)
            mask=~clean[hold]; denom=np.linalg.norm(Y[hold][mask],axis=1)+1e-12
            errs.append(float(np.mean(np.linalg.norm(p[mask]-Y[hold][mask],axis=1)/denom)))
        score=float(np.mean(errs)); print('alpha',alpha,'cv',score)
        if best is None or score<best[0]: best=(score,alpha)
    alpha=best[1]; beta=fit_ridge(tr,alpha)
    def pred(idx): return np.c_[Xn[idx],np.ones(idx.sum())]@beta
    def rel(idx):
        p=pred(idx); y=Y[idx]; return float(np.mean(np.linalg.norm(p-y,axis=1)/(np.linalg.norm(y,axis=1)+1e-12)))
    res={'alpha':float(alpha),'cv':best[0],'use_visual':args.use_visual,'clean_weight':args.clean_weight,'feature_dim':X.shape[1],'fit_velocity_rel':rel(tr_shift),'val_velocity_rel':rel(va_shift),'clean_pred_norm_val':float(np.mean(np.linalg.norm(pred((roles=='source_validation')&clean),axis=1))),'mu':mu.tolist(),'sd':sd.tolist(),'beta':beta.tolist()}
    args.out.write_text(json.dumps(res,indent=2)+'\n'); print(json.dumps({k:v for k,v in res.items() if k not in ('mu','sd','beta')},indent=2))

if __name__=='__main__': main()
