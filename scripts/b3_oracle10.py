import json,numpy as np
rows=json.load(open('/hy-tmp/patch-diag/runs/patch_diag/b3_flow_pairs_10.json'))['rows']; Y=np.array([r['r'] for r in rows]); S=np.array([r['state_id'] for r in rows]); states=sorted(set(S.tolist())); pilot={34,43,45}; unseen=[s for s in states if s not in pilot]
def feat(r,k):
 b=np.r_[r['h'],r['state8'],r['tau'],r['k']/12]
 if k=='M0': return b
 if k=='M1a': return np.r_[b,r['z_agent']]
 if k=='M1w': return np.r_[b,r['z_eye']]
 return np.r_[b,r['z_agent'],r['z_eye']]
def ridge(X,Y,a=100):
 mu=X.mean(0); sd=X.std(0)+1e-8; Z=(X-mu)/sd; D=np.c_[Z,np.ones(len(Z))]; P=a*np.eye(D.shape[1]); P[-1,-1]=0; return mu,sd,np.linalg.solve(D.T@D+P,D.T@Y)
def met(p,y):
 rel=float(np.mean(np.linalg.norm(p-y,axis=1)/(np.linalg.norm(y,axis=1)+1e-9))); agg=float(np.sqrt(np.sum((p-y)**2)/(np.sum(y**2)+1e-12))); cos=float(np.mean(np.sum(p*y,axis=1)/(np.linalg.norm(p,axis=1)*np.linalg.norm(y,axis=1)+1e-9))); mag=float(np.mean(np.linalg.norm(p,axis=1))/(np.mean(np.linalg.norm(y,axis=1))+1e-12)); return {'rel':rel,'agg':agg,'cos':cos,'mag':mag,'G_row':1-rel,'G_agg':1-agg}
out={}
for kind in ['M0','M1a','M1w','M1both']:
 X=np.stack([feat(r,kind) for r in rows]); sm={}; bm={}; cm={}
 for s in states:
  tr=S!=s; te=S==s; mu,sd,b=ridge(X[tr],Y[tr]); p=np.c_[((X[te]-mu)/sd),np.ones(te.sum())]@b; y=Y[te]; sm[str(s)]=met(p,y); c=y.mean(0)[None].repeat(len(y),0); cm[str(s)]=met(c,y)
  beta=np.clip(np.sum(p*y)/(np.sum(p*p)+1e-12),0,1); bm[str(s)]=met(beta*p,y)
 out[kind]={'states':sm,'constant':cm,'calibrated':bm}
def group(d,ids): return {k:float(np.mean([d[str(s)][k] for s in ids])) for k in ('rel','agg','cos','mag','G_row','G_agg')}
for kind in out:
 out[kind]['all']=group(out[kind]['states'],states); out[kind]['unseen7']=group(out[kind]['states'],unseen); out[kind]['pilot3']=group(out[kind]['states'],pilot); out[kind]['cal_all']=group(out[kind]['calibrated'],states); out[kind]['const_all']=group(out[kind]['constant'],states)
out['zero']=met(np.zeros_like(Y),Y); out['states']=states; out['unseen7']=unseen; out['pilot3']=sorted(pilot)
open('/hy-tmp/patch-diag/runs/patch_diag/b3_oracle10.json','w').write(json.dumps(out,indent=2)+'\n'); print(json.dumps({k:out[k]['all'] for k in ['M0','M1a','M1w','M1both']},indent=2)); print(json.dumps({'zero':out['zero'],'unseen_M1w':out['M1w']['unseen7'],'cal_M1w':out['M1w']['cal_all'],'const_M1w':out['M1w']['const_all']},indent=2))
