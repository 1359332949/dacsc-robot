"""R0-2: feature ablations, kNN identifiability, proper grouped CV."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    d=np.load(args.data); X=d['features'].astype(np.float64); Y=d['labels'].astype(np.float64); roles=d['roles']; states=d['states']; clean=d['clean'].astype(bool); hs=d['hs'].astype(np.float64)
    cfg={'B0_h':np.r_[768:1546],'B1_visual':np.r_[0:768,1536:1546],'B2_visual_h':np.r_[0:1546]}
    tr=(roles=='source_fit'); va=(roles=='source_validation'); tr_shift=tr&~clean; va_shift=va&~clean
    def ridge_fit(Xtr,Ytr,w,alpha):
        mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-8]=1; Z=(Xtr-mu)/sd; D=np.c_[Z,np.ones(len(Z))]
        A=D.T@(w[:,None]*D); A[-1,-1]-=alpha  # do not penalize intercept; undo prior? construct below
        pen=alpha*np.eye(D.shape[1]); pen[-1,-1]=0
        beta=np.linalg.solve(D.T@(w[:,None]*D)+pen,D.T@(w[:,None]*Ytr))
        return mu,sd,beta
    def pred(X,mu,sd,beta): return (np.c_[((X-mu)/sd),np.ones(len(X))]@beta)
    def metrics(p,y):
        rel=np.linalg.norm(p-y,axis=1)/(np.linalg.norm(y,axis=1)+1e-12); agg=float(np.sqrt(np.sum((p-y)**2)/(np.sum(y**2)+1e-12)))
        cos=np.sum(p*y,axis=1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(y,axis=1)+1e-12)
        return {'rel_mean':float(rel.mean()),'aggregate':agg,'cos_mean':float(cos.mean()),'n':int(len(y))}
    out={'schema':'r0_analysis_v1','ablations':{},'knn':{},'variance':{}}
    for name,cols in cfg.items():
        Xs=X[:,cols]; uniq=sorted(set(states[tr])); alphas=(1,10,100,1000,10000); best=None
        for a in alphas:
            es=[]
            for s in uniq:
                hold=tr&(states==s); train=tr&(states!=s); mu,sd,beta=ridge_fit(Xs[train],Y[train],np.where(clean[train],1.0,1.0),a)
                p=pred(Xs[hold],mu,sd,beta); m=~clean[hold]; es.append(metrics(p[m],Y[hold][m])['rel_mean'])
            score=float(np.mean(es)); best=(score,a) if best is None or score<best[0] else best
        a=best[1]; mu,sd,beta=ridge_fit(Xs[tr],Y[tr],np.where(clean[tr],1.0,1.0),a)
        p=pred(Xs[va_shift],mu,sd,beta); pclean=pred(Xs[va&clean],mu,sd,beta)
        # hidden ratio for clean shift/W pinv is not available here; report residual norm
        out['ablations'][name]={'alpha':int(a),'cv_rel':best[0],'val':metrics(p,Y[va_shift]),'clean_pred_norm':float(np.mean(np.linalg.norm(pclean,axis=1)))}
    # kNN on B2 features, fit->validation shift
    cols=cfg['B2_visual_h']; mu=X[tr][:,cols].mean(0); sd=X[tr][:,cols].std(0); sd[sd<1e-8]=1; A=(X[tr_shift][:,cols]-mu)/sd; B=(X[va_shift][:,cols]-mu)/sd
    # subsample fit for memory
    rng=np.random.default_rng(0); idx=rng.choice(len(A),min(2000,len(A)),replace=False); A2=A[idx]; Yfit=Y[tr_shift][idx]
    preds=[]
    for i in range(len(B)):
        d2=np.sum((A2-B[i])**2,axis=1); preds.append(Yfit[int(np.argmin(d2))])
    out['knn']=metrics(np.array(preds),Y[va_shift])
    # variance decomposition by state (balanced sample count assumed)
    ys=Y[tr_shift]; ss=states[tr_shift]; means=[]; wv=[]
    for s in sorted(set(ss)):
        g=ys[ss==s]; means.append(g.mean(0)); wv.append(np.mean(np.sum((g-g.mean(0))**2,axis=1)))
    means=np.array(means); total=float(np.mean(np.sum((ys-ys.mean(0))**2,axis=1))); within=float(np.mean(wv)); between=float(np.mean(np.sum((means-ys.mean(0))**2,axis=1)))
    out['variance']={'between_state_mean_sq':between,'within_mean_sq':within,'total_mean_sq':total,'between_share':between/(total+1e-12),'within_share':within/(total+1e-12)}
    args.out.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
