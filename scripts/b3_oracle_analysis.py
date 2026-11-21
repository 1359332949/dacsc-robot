import json,numpy as np
rows=json.load(open('/hy-tmp/patch-diag/runs/patch_diag/b3_flow_pairs.json'))['rows']
def feats(r,vis):
    b=np.r_[r['h'],r['state8'],r['tau'],r['k']/12]
    return np.r_[b,r['visual']] if vis else b
Y=np.array([r['r'] for r in rows]); S=np.array([r['state_id'] for r in rows])
def ridge(X,Y,a=100):
    mu=X.mean(0); sd=X.std(0)+1e-8; Z=(X-mu)/sd; D=np.c_[Z,np.ones(len(Z))]; P=a*np.eye(D.shape[1]); P[-1,-1]=0; return mu,sd,np.linalg.solve(D.T@D+P,D.T@Y)
def metrics(p,y):
    rel=float(np.mean(np.linalg.norm(p-y,axis=1)/(np.linalg.norm(y,axis=1)+1e-9))); agg=float(np.sqrt(np.sum((p-y)**2)/(np.sum(y**2)+1e-12))); cos=float(np.mean(np.sum(p*y,axis=1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(y,axis=1)+1e-9))); mag=float(np.mean(np.linalg.norm(p,axis=1))/(np.mean(np.linalg.norm(y,axis=1))+1e-12)); return rel,agg,cos,mag
out={}
for name,vis in [('M0',False),('M1',True)]:
    X=np.stack([feats(r,vis) for r in rows]); ms=[]
    for s in sorted(set(S)):
        tr=S!=s; te=S==s; mu,sd,b=ridge(X[tr],Y[tr]); p=np.c_[((X[te]-mu)/sd),np.ones(te.sum())]@b; ms.append(metrics(p,Y[te]))
    out[name]={'n':len(rows),'state_metrics':{str(s):m for s,m in zip(sorted(set(S)),ms)},'mean':[float(np.mean([m[i] for m in ms])) for i in range(4)]}
# zero baseline
out['zero']=metrics(np.zeros_like(Y),Y)
open('/hy-tmp/patch-diag/runs/patch_diag/b3_oracle_analysis.json','w').write(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
