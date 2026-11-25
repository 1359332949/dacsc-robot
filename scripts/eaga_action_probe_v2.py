"""EAGA action probe v2: single preprocessing, true weighted moments, clean-relative and frozen-delta geometry."""
from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
from corp import load_policy
from corp.preprocessing import prepare_observation
from dacsc.single_block_probe import align,score
from target_only_action_validation import sample_z,visual_from_obs
INPUT=ROOT/'runs/invariance_sensitivity_v1b/input'; OUT=ROOT/'runs/eaga_probe_v1'
policy=None

def raw_row(r):
    z=np.load(INPUT/(r['id']+'.npz'),allow_pickle=False)
    return {'images':{v:torch.from_numpy(z[v].copy())[None] for v in ('agentview','eye_in_hand')},'state':torch.from_numpy(z['state'].copy()),'instruction':[r['lang']]}
def prep(raw): return prepare_observation(raw,config=policy.config,normalizer=policy.normalizer,device='cuda:0')
def attention(po):
    ex=policy.representation_extractor; rs=ex.cross_attention_resampler; p=ex.visual_encoder(torch.cat([po['images'][v] for v in ex.view_names])); B=po['state'].shape[0]; ws=[]
    for i,v in enumerate(ex.view_names):
        pt=p[i*B:(i+1)*B]; lang,pool=ex.language_encoder(po['instruction']); gamma,beta=rs.film(pool).unsqueeze(1).chunk(2,dim=-1); q=rs.queries[i].unsqueeze(0).expand(B,-1,-1); q=q*(1+gamma)+beta; patches=rs.input_projection(pt)+rs.view_embedding[i].unsqueeze(0)
        _,w=rs.cross_attention(q,patches,patches,need_weights=True,average_attn_weights=False); ws.append(w.mean(1).mean(1)[0])
    return torch.stack(ws)
def wtil(w):
    w=.8*w+.2/256; return w/w.sum(-1,keepdim=True)
def align_weighted(x,ref,w,alpha=.75):
    mu,ss,tau=ref; p=x[:,1:]; ww=wtil(w); m=(ww[:,:,None]*p).sum(1); sd=((ww[:,:,None]*(p-m[:,None,:]).square()).sum(1)+1e-6).sqrt(); active=score(m,sd,mu,ss)>tau; gain=(((1-alpha)*sd+alpha*ss)/(sd.square()+1e-6).sqrt()).clamp(.5,2); out=torch.where(active[:,None,:],gain[:,None,:]*(p-m[:,None,:])+((1-alpha)*m+alpha*mu)[:,None,:],p); return torch.cat((x[:,:1],out),1)
def align_spatial(x,ref,w,pref=.35,bg=.75):
    mu,ss,tau=ref; p=x[:,1:]; m=p.mean(1); sd=p.std(1,unbiased=False); active=score(m,sd,mu,ss)>tau; wn=(w/w.amax(-1,keepdim=True)).clamp(0,1); a=(pref+(bg-pref)*(1-wn)).clamp(pref,bg); base=(((1-.75)*sd+.75*ss)/(sd.square()+1e-6).sqrt()).clamp(.5,2); a3=a[:,:,None]; g=a3*base[:,None,:]+(1-a3); out=torch.where(active[:,None,:],g*(p-m[:,None,:])+((1-a3)*m[:,None,:]+a3*mu[:,None,:]),p); return torch.cat((x[:,:1],out),1)
def predict(raw,z,arm,ref,w=None):
    h=None
    if arm=='v1': h=policy.representation_extractor.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align(x,ref,.75,'dead'))
    elif arm=='weighted': h=policy.representation_extractor.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align_weighted(x,ref,w,.75))
    elif arm=='spatial': h=policy.representation_extractor.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align_spatial(x,ref,w))
    try:
        _,rep=visual_from_obs(policy,raw); return policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu()
    finally:
        if h: h.remove()
def dist(a,b):
    d=(a-b).abs(); return float(d[:,:3].mean()),float(d[:,3:6].mean())
def main():
    global policy
    policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False)
    rows=[]
    for f in sorted(INPUT.glob('index_*.json')): rows+=json.load(open(f))
    t0=[r for r in rows if not r.get('geometry')]; moves=[r for r in rows if r.get('geometry')]
    raw={r['id']:raw_row(r) for r in rows}; po={r['id']:prep(raw[r['id']]) for r in t0}; w={r['id']:attention(po[r['id']]) for r in t0}
    for r in moves: w[r['id']]=attention(prep(raw[r['id']]))
    clean={r['task']:r for r in t0 if r['factor']=='clean'}; assert len(clean)==10
    full=np.load(ROOT/'runs/full_traj_v1/full.npz',allow_pickle=False); uniform={t:tuple(torch.as_tensor(full[f'{t}__3__{k}'],device='cuda:0') for k in ('mean','std','tau')) for t in clean}
    # weighted source refs from 9 other tasks, using each Clean frame's own t0 attention
    source={f.stem:np.load(f) for f in (ROOT/'runs/full_traj_v1/source').glob('*.npy')}
    weighted={}
    for t in clean:
        ms=[]; qs=[]; ws=[]
        for u in clean:
            if u==t: continue
            arr=torch.from_numpy(source[clean[u]['id']]).float(); W=wtil(w[clean[u]['id']].double()).cpu(); ms.append(arr[:,:,0,:].mean(0)); qs.append(arr[:,:,1,:].mean(0)); ws.append(W)
        M=torch.stack(ms).mean(0); Q=torch.stack(qs).mean(0); W=torch.stack(ws).mean(0)
        mu=(M[:,None,:]*W[:,:,None]).sum(1)/W.sum(1)[:,None]; var=((Q-M.square())[:,None,:]*W[:,:,None]).sum(1)/W.sum(1)[:,None]
        weighted[t]=(mu.float().cuda(),var.clamp_min(0).sqrt().float().cuda(),uniform[t][2].float())
    # 4 fixed noises, same for all arms
    zs=[torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(s)) for s in (11,22,33,44)]
    # clean frozen references per task per z
    clean_ref={}
    for t,r in clean.items():
        clean_ref[t]=[predict(raw[r['id']],z,'frozen',None) for z in zs]
    # t0 relative-to-clean metrics
    res=[]
    for r in t0:
        for zi,z in enumerate(zs):
            frozen=predict(raw[r['id']],z,'frozen',None); base_xyz,base_rot=dist(frozen,clean_ref[r['task']][zi])
            for arm in ('v1','weighted','spatial'):
                ref=uniform[r['task']] if arm=='v1' else weighted[r['task']]; a=predict(raw[r['id']],z,arm,ref,w[r['id']] if arm!='v1' else None); adx,adr=dist(a,clean_ref[r['task']][zi]); res.append({'id':r['id'],'task':r['task'],'factor':r['factor'],'sub':r['sub'],'arm':arm,'z':zi,'near_xyz':(base_xyz-adx)/(base_xyz+1e-12)*100 if r['factor']!='clean' else 0.0,'near_rot':(base_rot-adr)/(base_rot+1e-12)*100 if r['factor']!='clean' else 0.0,'clean_drift_xyz':float((a-frozen).abs()[:,:3].mean()),'clean_drift_rot':float((a-frozen).abs()[:,3:6].mean())})
    # geometry: adapt delta vs frozen delta
    geom=[]
    caseids=sorted({r['case'] for r in moves if r.get('factor')=='clean'})
    for caseid in caseids:
        zr=next(r for r in moves if r['case']==caseid and np.allclose(r.get('offset_m',[9,9,9]),[0,0,0]))
        nz=[r for r in moves if r['case']==caseid and not np.allclose(r.get('offset_m',[9,9,9]),[0,0,0])]
        task=zr['task']
        for r in nz:
            for zi,z in enumerate(zs):
                f0=predict(raw[zr['id']],z,'frozen',None); f1=predict(raw[r['id']],z,'frozen',None); df=(f1-f0).numpy()[:,:3].mean(0)
                for arm in ('v1','weighted','spatial'):
                    ref=uniform[task] if arm=='v1' else weighted[task]; a0=predict(raw[zr['id']],z,arm,ref,w[zr['id']] if arm!='v1' else None); a1=predict(raw[r['id']],z,arm,ref,w[r['id']] if arm!='v1' else None); da=(a1-a0).numpy()[:,:3].mean(0)
                    cos=float(np.dot(da,df)/(np.linalg.norm(da)*np.linalg.norm(df)+1e-12)); ratio=float(np.linalg.norm(da)/(np.linalg.norm(df)+1e-12)); geom.append({'arm':arm,'case':task,'cos':cos,'ratio':ratio})
    # frozen geometry is identity by construction; include self check
    for r in moves:
        pass
    groups={}
    for fac in ['clean','sensor_noise','background','lighting']:
        for arm in ('v1','weighted','spatial'):
            sub=[x for x in res if x['factor']==fac and x['arm']==arm]; groups[f'{fac}__{arm}']={'n':len(sub),'near_xyz':float(np.mean([x['near_xyz'] for x in sub])),'near_rot':float(np.mean([x['near_rot'] for x in sub])),'drift_xyz':float(np.mean([x['clean_drift_xyz'] for x in sub])),'drift_rot':float(np.mean([x['clean_drift_rot'] for x in sub]))}
    for arm in ('v1','weighted','spatial'):
        sub=[x for x in geom if x['arm']==arm]; groups[f'geom__{arm}']={'n':len(sub),'cos_median':float(np.median([x['cos'] for x in sub])),'ratio_median':float(np.median([x['ratio'] for x in sub]))}
    # identity check on clean frozen vs itself should be zero; report V1 clean drift
    OUT.mkdir(exist_ok=True); (OUT/'actions_v2.json').write_text(json.dumps({'groups':groups,'rows':res,'geometry':geom},indent=2)+'\n'); print(json.dumps(groups,indent=2))
if __name__=='__main__': main()
