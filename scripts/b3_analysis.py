"""B3-1 pilot: P0/P1 transition prediction on boundary rows."""
import json,numpy as np
from pathlib import Path
rows=json.loads(Path('/hy-tmp/patch-diag/runs/patch_diag/b3_boundary.json').read_text())['rows']
def feats(r,visual):
    acts=np.zeros((12,7),np.float64); mask=np.zeros(12,np.float64); a=np.array(r['actions_executed']); acts[:len(a)]=a; mask[:len(a)]=1; A=np.r_[acts.reshape(-1),mask]
    base=np.r_[r['state8_start'],A]
    if not visual: return base
    va=np.array(r['view_agent_start']).mean(0); ve=np.array(r['view_eye_start']).mean(0); return np.r_[base,va,ve]
def ridge(X,Y,alpha=10.0):
    mu=X.mean(0); sd=X.std(0)+1e-8; Z=(X-mu)/sd; D=np.c_[Z,np.ones(len(Z))]; P=alpha*np.eye(D.shape[1]); P[-1,-1]=0; return mu,sd,np.linalg.solve(D.T@D+P,D.T@Y)
def pred(X,m,s,b): return np.c_[((X-m)/s),np.ones(len(X))]@b
def met(p,y): return float(np.mean(np.linalg.norm(p-y,axis=1)/(np.linalg.norm(y,axis=1)+1e-9)))
clean=[r for r in rows if r['domain']=='clean']; shift=[r for r in rows if r['domain']=='shift']
out={}
for name,vis in [('P0',False),('P1',True)]:
    Xc=np.stack([feats(r,vis) for r in clean]); Yc=np.stack([r['dstate8'] for r in clean]); Xs=np.stack([feats(r,vis) for r in shift]); Ys=np.stack([r['dstate8'] for r in shift]); m,s,b=ridge(Xc,Yc); out[name]={'train_clean':met(pred(Xc,m,s,b),Yc),'test_shift':met(pred(Xs,m,s,b),Ys)}
print(json.dumps(out,indent=2))
