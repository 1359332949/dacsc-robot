import json,numpy as np
rows=json.load(open('/hy-tmp/patch-diag/runs/patch_diag/b3_flow_pairs_v2.json'))['rows']
def feat(r,kind):
    b=np.r_[r['h'],r['state8'],r['tau'],r['k']/12]
    if kind=='M0': return b
    if kind=='M1a': return np.r_[b,r['z_agent']]
    if kind=='M1w': return np.r_[b,r['z_eye']]
    return np.r_[b,r['z_agent'],r['z_eye']]
Y=np.array([r['r'] for r in rows]); S=np.array([r['state_id'] for r in rows])
def ridge(X,Y,a=100):
    mu=X.mean(0); sd=X.std(0)+1e-8; Z=(X-mu)/sd; D=np.c_[Z,np.ones(len(Z))]; P=a*np.eye(D.shape[1]); P[-1,-1]=0; return mu,sd,np.linalg.solve(D.T@D+P,D.T@Y)
def met(p,y):
    rel=float(np.mean(np.linalg.norm(p-y,axis=1)/(np.linalg.norm(y,axis=1)+1e-9))); agg=float(np.sqrt(np.sum((p-y)**2)/(np.sum(y**2)+1e-12))); cos=float(np.mean(np.sum(p*y,axis=1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(y,axis=1)+1e-9))); mag=float(np.mean(np.linalg.norm(p,axis=1))/(np.mean(np.linalg.norm(y,axis=1))+1e-12)); G=float(np.mean((np.linalg.norm(y,axis=1)-np.linalg.norm(p-y,axis=1))/(np.linalg.norm(y,axis=1)+1e-9))); return {'rel':rel,'agg':agg,'cos':cos,'mag':mag,'G':G}
out={}
for kind in ['M0','M1a','M1w','M1both']:
    X=np.stack([feat(r,kind) for r in rows]); ms={}
    for s in sorted(set(S)):
        tr=S!=s; te=S==s; mu,sd,b=ridge(X[tr],Y[tr]); p=np.c_[((X[te]-mu)/sd),np.ones(te.sum())]@b; ms[str(int(s))]=met(p,Y[te])
    out[kind]={'states':ms,'mean':{k:float(np.mean([m[k] for m in ms.values()])) for k in ('rel','agg','cos','mag','G')}}
out['zero']=met(np.zeros_like(Y),Y)
open('/hy-tmp/patch-diag/runs/patch_diag/b3_oracle2.json','w').write(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
