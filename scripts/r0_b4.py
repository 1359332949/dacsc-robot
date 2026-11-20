"""R0-2B4: fold-internal learned query (B4a) and low-rank QKV (B4b)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F

class Retriever(nn.Module):
    def __init__(self,kind,qdim=32,vrank=32):
        super().__init__(); self.kind=kind; self.ln=nn.LayerNorm(768); self.pq=nn.Linear(768,(768 if kind=='b4a' else qdim),bias=False)
        if kind=='b4b': self.pk=nn.Linear(768,qdim,bias=False); self.pv=nn.Linear(768,vrank,bias=False)
        self.head=nn.Linear((1536 if kind=='b4a' else 64)+768+8+2,7)
    def forward(self,tokens,h,state,tau,k):
        B=tokens.shape[0]; q=F.normalize(self.pq(self.ln(h)),dim=-1)[:,None,:]
        outs=[]
        for v in range(2):
            Z=tokens[:,v]; Z=self.ln(Z)
            if self.kind=='b4a': K=V=F.normalize(Z,dim=-1)
            else: K=F.normalize(self.pk(Z),dim=-1); V=self.pv(Z)
            s=(q@K.transpose(1,2)).squeeze(1)/(K.shape[-1]**0.5); a=s.softmax(-1); x=(a[:,None,:]@V).squeeze(1)
            outs.append(x)
        f=torch.cat([outs[0],outs[1],h,state,tau[:,None],k[:,None]],dim=1)
        return self.head(f)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tokens',type=Path,required=True); ap.add_argument('--rgb',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--epochs',type=int,default=12); ap.add_argument('--batch',type=int,default=128); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); d=np.load(args.tokens); z=np.load(args.rgb); man=json.loads(Path(str(args.rgb)+'.manifest.json').read_text()); s8={r['state_id']:z['state8'][i] for i,r in enumerate(man['records'])}
    T=d['tokens']; H=d['hs']; Y=d['labels']; roles=d['roles']; states=d['states']; clean=d['clean'].astype(bool); tau=d['tau']; k=d['k']; S=np.stack([s8[int(s)] for s in states]).astype(np.float32)
    dev=torch.device(args.device); fit_ids=sorted(set(states[roles=='source_fit'])); val_ids=sorted(set(states[roles=='source_validation']))
    def metrics(pred,y):
        denom=np.linalg.norm(y,axis=1)+1e-12; rel=np.linalg.norm(pred-y,axis=1)/denom; cos=np.sum(pred*y,axis=1)/(np.linalg.norm(pred,axis=1)*denom+1e-12); agg=float(np.sqrt(np.sum((pred-y)**2)/(np.sum(y**2)+1e-12))); mag=float(np.linalg.norm(pred,axis=1).mean()/(np.linalg.norm(y,axis=1).mean()+1e-12)); return float(rel.mean()),float(cos.mean()),agg,mag
    results={'b4a':[],'b4b':[]}
    for kind in ('b4a','b4b'):
        for held in fit_ids:
            tr=(roles=='source_fit')&(states!=held); va=(roles=='source_fit')&(states==held)
            # per-fold normalization
            hmu=H[tr].mean(0); hsd=H[tr].std(0)+1e-8; tmu=T[tr].astype(np.float32).mean((0,1,2)); tsd=T[tr].astype(np.float32).std((0,1,2))+1e-8; smu=S[tr].mean(0); ssd=S[tr].std(0)+1e-8
            model=Retriever(kind).to(dev); opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4)
            idx=np.where(tr)[0]; rng=np.random.default_rng(0)
            Xh=torch.tensor(H[idx],dtype=torch.float32); Xt=torch.tensor(T[idx].astype(np.float32),dtype=torch.float32); Xs=torch.tensor(S[idx],dtype=torch.float32); Xtau=torch.tensor(tau[idx],dtype=torch.float32); Xk=torch.tensor(k[idx],dtype=torch.float32); Yt=torch.tensor(Y[idx],dtype=torch.float32)
            Xh=(Xh-torch.tensor(hmu))/torch.tensor(hsd); Xt=(Xt-torch.tensor(tmu))/torch.tensor(tsd); Xs=(Xs-torch.tensor(smu))/torch.tensor(ssd)
            for ep in range(args.epochs):
                perm=rng.permutation(len(idx))
                for b in range(0,len(idx),args.batch):
                    ii=torch.tensor(perm[b:b+args.batch]); y=Yt[ii].to(dev); pred=model(Xt[ii].to(dev),Xh[ii].to(dev),Xs[ii].to(dev),Xtau[ii].to(dev),Xk[ii].to(dev)); loss=((pred-y)**2).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            with torch.no_grad():
                vidx=np.where(va)[0]; pred=model(((torch.tensor(T[vidx].astype(np.float32),dtype=torch.float32)-torch.tensor(tmu))/torch.tensor(tsd)).to(dev),((torch.tensor(H[vidx],dtype=torch.float32)-torch.tensor(hmu))/torch.tensor(hsd)).to(dev),((torch.tensor(S[vidx],dtype=torch.float32)-torch.tensor(smu))/torch.tensor(ssd)).to(dev),torch.tensor(tau[vidx],dtype=torch.float32).to(dev),torch.tensor(k[vidx],dtype=torch.float32).to(dev)).cpu().numpy()
            cm=np.array(clean[vidx]); shift_mask=~cm; y=Y[vidx][shift_mask]; rel,cos,agg,mag=metrics(pred[shift_mask],y); cpred=pred[cm]; cnorm=float(np.mean(np.linalg.norm(cpred,axis=1))) if cm.any() else None
            results[kind].append({'held_state':int(held),'n':int(len(vidx)),'rel':rel,'cos':cos,'agg':agg,'mag_ratio':mag,'clean_pred_norm':cnorm})
            print(json.dumps({'kind':kind,'held':int(held),'rel':rel,'cos':cos,'clean':cnorm}),flush=True)
    # validation one-shot: train on all fit states
    for kind in ('b4a','b4b'):
        tr=(roles=='source_fit'); va=(roles=='source_validation'); hmu=H[tr].mean(0); hsd=H[tr].std(0)+1e-8; tmu=T[tr].astype(np.float32).mean((0,1,2)); tsd=T[tr].astype(np.float32).std((0,1,2))+1e-8; smu=S[tr].mean(0); ssd=S[tr].std(0)+1e-8
        model=Retriever(kind).to(dev); opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4); idx=np.where(tr)[0]; rng=np.random.default_rng(0)
        Xh=((torch.tensor(H[idx],dtype=torch.float32)-torch.tensor(hmu))/torch.tensor(hsd)).to(dev); Xt=((torch.tensor(T[idx].astype(np.float32),dtype=torch.float32)-torch.tensor(tmu))/torch.tensor(tsd)).to(dev); Xs=((torch.tensor(S[idx],dtype=torch.float32)-torch.tensor(smu))/torch.tensor(ssd)).to(dev); Xtau=torch.tensor(tau[idx],dtype=torch.float32).to(dev); Xk=torch.tensor(k[idx],dtype=torch.float32).to(dev); Yt=torch.tensor(Y[idx],dtype=torch.float32).to(dev)
        for ep in range(args.epochs):
            for b in range(0,len(idx),args.batch):
                ii=torch.tensor(rng.permutation(len(idx))[b:b+args.batch]); pred=model(Xt[ii],Xh[ii],Xs[ii],Xtau[ii],Xk[ii]); loss=((pred-Yt[ii])**2).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        with torch.no_grad():
            vidx=np.where(va)[0]; pred=model(((torch.tensor(T[vidx].astype(np.float32),dtype=torch.float32)-torch.tensor(tmu))/torch.tensor(tsd)).to(dev),((torch.tensor(H[vidx],dtype=torch.float32)-torch.tensor(hmu))/torch.tensor(hsd)).to(dev),((torch.tensor(S[vidx],dtype=torch.float32)-torch.tensor(smu))/torch.tensor(ssd)).to(dev),torch.tensor(tau[vidx],dtype=torch.float32).to(dev),torch.tensor(k[vidx],dtype=torch.float32).to(dev)).cpu().numpy()
        cm=np.array(clean[vidx]); shift_mask=~cm; y=Y[vidx][shift_mask]; rel,cos,agg,mag=metrics(pred[shift_mask],y); results[kind+'_val']={'rel':rel,'cos':cos,'agg':agg,'mag_ratio':mag,'n':len(vidx)}
    args.out.write_text(json.dumps({'results':results,'epochs':args.epochs,'lr':args.lr},indent=2)+'\n'); print(json.dumps(results,indent=2))

if __name__=='__main__': main()
