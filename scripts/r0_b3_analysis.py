"""R0-2B3: per-view pooling vs fixed low-rank action-conditioned token retrieval."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tokens',type=Path,required=True); ap.add_argument('--rgb',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    d=np.load(args.tokens); z=np.load(args.rgb); man=json.loads(Path(str(args.rgb)+'.manifest.json').read_text())
    # state8 per state
    s8={r['state_id']:z['state8'][i] for i,r in enumerate(man['records'])}
    T=d['tokens'].astype(np.float32); H=d['hs'].astype(np.float64); Y=d['labels'].astype(np.float64); roles=d['roles']; states=d['states']; clean=d['clean'].astype(bool); tau=d['tau'].astype(np.float64); k=d['k'].astype(np.float64)
    S=np.stack([s8[int(s)] for s in states]).astype(np.float64)
    # fixed low-rank retrieval
    rng=np.random.default_rng(20261120); dim=768; qdim=32; vdim=64
    Wq=rng.normal(0,1/np.sqrt(dim),(dim,qdim)); Wk=rng.normal(0,1/np.sqrt(dim),(dim,qdim)); Wv=rng.normal(0,1/np.sqrt(dim),(dim,vdim)); Wo=rng.normal(0,1/np.sqrt(vdim),(vdim,dim))
    q=H@Wq; K=(T@Wk).reshape(len(T),-1,qdim); V=(T@Wv).reshape(len(T),-1,vdim); scores=np.einsum('bd,bnd->bn',q,K)/np.sqrt(qdim); scores-=scores.max(axis=-1,keepdims=True); A=np.exp(scores); A/=A.sum(-1,keepdims=True); X=np.einsum('bn,bnd->bd',A,V)@Wo
    viewmean=T.mean(axis=2)  # [N,2,768]
    def ridge_cv(Xf):
        tr=(roles=='source_fit'); va=(roles=='source_validation'); tr_shift=tr&~clean; va_shift=va&~clean; uniq=sorted(set(states[tr])); best=None
        def fit(idx,a):
            mu=Xf[idx].mean(0); sd=Xf[idx].std(0); sd[sd<1e-8]=1; Z=(Xf[idx]-mu)/sd; D=np.c_[Z,np.ones(len(Z))]; w=np.ones(len(Z)); pen=a*np.eye(D.shape[1]); pen[-1,-1]=0
            beta=np.linalg.solve(D.T@D+pen,D.T@Y[idx]); return mu,sd,beta
        def pred(idx,mu,sd,beta): return np.c_[((Xf[idx]-mu)/sd),np.ones(idx.sum())]@beta
        for a in (1,10,100,1000,10000,100000):
            es=[]
            for s in uniq:
                hold=tr&(states==s); train=tr&(states!=s); mu,sd,beta=fit(train,a); p=pred(hold,mu,sd,beta); m=~clean[hold]; y=Y[hold][m]; es.append(float(np.mean(np.linalg.norm(p[m]-y,axis=1)/(np.linalg.norm(y,axis=1)+1e-12))))
            sc=float(np.mean(es)); best=(sc,a) if best is None or sc<best[0] else best
        mu,sd,beta=fit(tr,best[1]); p=pred(va_shift,mu,sd,beta); y=Y[va_shift]; rel=float(np.mean(np.linalg.norm(p-y,axis=1)/(np.linalg.norm(y,axis=1)+1e-12))); agg=float(np.sqrt(np.sum((p-y)**2)/(np.sum(y**2)+1e-12))); cos=float(np.mean(np.sum(p*y,axis=1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(y,axis=1)+1e-12)))
        pc=pred(va&clean,mu,sd,beta); # hidden ratio requires W; skip here
        return {'alpha':best[1],'cv_rel':best[0],'val_rel':rel,'val_agg':agg,'val_cos':cos,'clean_pred_norm':float(np.mean(np.linalg.norm(pc,axis=1)))}
    Xb0=np.c_[H,S,tau[:,None],k[:,None]]; Xb3a=np.c_[viewmean.reshape(len(T),-1),H,S,tau[:,None],k[:,None]]; Xb3b=np.c_[H,X,S,tau[:,None],k[:,None]]
    out={'B0':ridge_cv(Xb0),'B3a':ridge_cv(Xb3a),'B3b':ridge_cv(Xb3b),'n':int(len(T)),'feature_dims':{'B0':Xb0.shape[1],'B3a':Xb3a.shape[1],'B3b':Xb3b.shape[1]}}
    args.out.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
