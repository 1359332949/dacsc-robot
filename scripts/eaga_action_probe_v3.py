"""EAGA probe v3: true t=0 patch references, LOTO weighted tau, 5 arms."""
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
ARMS=('v1_full','spatial_full','v1_t0','weighted_t0','spatial_t0')

def raw_row(r):
    z=np.load(INPUT/(r['id']+'.npz'),allow_pickle=False)
    return {'images':{v:torch.from_numpy(z[v].copy())[None] for v in ('agentview','eye_in_hand')},'state':torch.from_numpy(z['state'].copy()),'instruction':[r['lang']]}
def prep(raw): return prepare_observation(raw,config=policy.config,normalizer=policy.normalizer,device='cuda:0')
def wtil(w):
    w=.8*w+.2/256; return w/w.sum(-1,keepdim=True)
def source_patch_and_attention(po):
    ex=policy.representation_extractor; rs=ex.cross_attention_resampler; cap={}
    def hook(m,a,x): cap['x']=x[:,1:].detach(); return None
    h=ex.visual_encoder.backbone.blocks[2].register_forward_hook(hook)
    try: p=ex.visual_encoder(torch.cat([po['images'][v] for v in ex.view_names]))
    finally: h.remove()
    B=po['state'].shape[0]; lang,pool=ex.language_encoder(po['instruction']); gamma,beta=rs.film(pool).unsqueeze(1).chunk(2,dim=-1); ws=[]
    for i,v in enumerate(ex.view_names):
        pt=p[i*B:(i+1)*B]; q=rs.queries[i].unsqueeze(0).expand(B,-1,-1); q=q*(1+gamma)+beta; patches=rs.input_projection(pt)+rs.view_embedding[i].unsqueeze(0)
        _,w=rs.cross_attention(q,patches,patches,need_weights=True,average_attn_weights=False); ws.append(w.mean(1).mean(1)[0])
    return cap['x'], torch.stack(ws)  # [2,256,384], [2,256]

def fit_weighted(patch,w):
    ww=wtil(w.double()); m=(ww[:,:,None]*patch.double()).sum(1); q=(ww[:,:,None]*patch.double().square()).sum(1); return m,q
def ref_weighted(others):
    # others list of (m,q) per task; outer 9, inner 8+1 LOTO tau
    M=torch.stack([x[0] for x in others]).float(); Q=torch.stack([x[1] for x in others]).float(); mu=M.mean(0); ss=(Q.mean(0)-mu.square()).clamp_min(0).sqrt(); ss=torch.clamp(ss,min=1e-3)
    scores=[]
    for i in range(len(others)):
        ins=[others[j] for j in range(len(others)) if j!=i]; mi=torch.stack([x[0] for x in ins]).mean(0); si=(torch.stack([x[1] for x in ins]).mean(0)-mi.square()).clamp_min(0).sqrt().clamp_min(1e-3); m,q=others[i]; scores.append(score(m.float(),(q-m.square()).clamp_min(0).sqrt().float(),mi,si))
    return mu,ss,torch.stack(scores).quantile(.95,dim=0)
def ref_unif(patches):
    # patches list [2,256,384]
    raw=torch.stack([torch.stack([p.mean(1),p.square().mean(1)]) for p in patches]) # [9,2mom,2view,384]
    mu=raw[:,0].mean(0); ss=(raw[:,1].mean(0)-mu.square()).clamp_min(0).sqrt().clamp_min(1e-3); scores=[]
    for i in range(len(patches)):
        others=[raw[j] for j in range(len(patches)) if j!=i]; ins=torch.stack(others); mi=ins[:,0].mean(0); si=(ins[:,1].mean(0)-mi.square()).clamp_min(0).sqrt().clamp_min(1e-3); m=raw[i,0]; q=raw[i,1]; scores.append(score(m,(q-m.square()).clamp_min(0).sqrt(),mi,si))
    return mu,ss,torch.stack(scores).quantile(.95,dim=0)
def align_weighted(x,ref,w,alpha=.75):
    mu,ss,tau=ref; p=x[:,1:]; ww=wtil(w); m=(ww[:,:,None]*p).sum(1); sd=((ww[:,:,None]*(p-m[:,None,:]).square()).sum(1)+1e-6).sqrt(); active=score(m,sd,mu,ss)>tau; gain=(((1-alpha)*sd+alpha*ss)/(sd.square()+1e-6).sqrt()).clamp(.5,2); out=torch.where(active[:,None,:],gain[:,None,:]*(p-m[:,None,:])+((1-alpha)*m+alpha*mu)[:,None,:],p); return torch.cat((x[:,:1],out),1)
def align_spatial(x,ref,w,pref=.35,bg=.75):
    mu,ss,tau=ref; p=x[:,1:]; m=p.mean(1); sd=p.std(1,unbiased=False); active=score(m,sd,mu,ss)>tau; wn=(w/w.amax(-1,keepdim=True)).clamp(0,1); a=(pref+(bg-pref)*(1-wn)).clamp(pref,bg); a3=a[:,:,None]; gain=(((1-a3)*sd[:,None,:]+a3*ss[:,None,:])/(sd[:,None,:].square()+1e-6).sqrt()).clamp(.5,2); out=torch.where(active[:,None,:],gain*(p-m[:,None,:])+((1-a3)*m[:,None,:]+a3*mu[:,None,:]),p); return torch.cat((x[:,:1],out),1)
def predict(raw,z,arm,ref,w=None):
    h=None
    if arm=='frozen':
        _,rep=visual_from_obs(policy,raw); return policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu()
    if arm=='v1_full' or arm=='v1_t0': h=policy.representation_extractor.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align(x,ref,.75,'dead'))
    elif arm=='spatial_full' or arm=='spatial_t0': h=policy.representation_extractor.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align_spatial(x,ref,w))
    elif arm=='weighted_t0': h=policy.representation_extractor.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align_weighted(x,ref,w))
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
    raw={r['id']:raw_row(r) for r in rows}
    po={r['id']:prep(raw[r['id']]) for r in t0}; patch={}; att={}
    for r in t0: patch[r['id']],att[r['id']]=source_patch_and_attention(po[r['id']])
    for r in moves:
        po[r['id']]=prep(raw[r['id']]); patch[r['id']],att[r['id']]=source_patch_and_attention(po[r['id']])
    clean={r['task']:r for r in t0 if r['factor']=='clean'}; assert len(clean)==10
    full=np.load(ROOT/'runs/full_traj_v1/full.npz',allow_pickle=False); ref_full={t:tuple(torch.as_tensor(full[f'{t}__3__{k}'],device='cuda:0').float() for k in ('mean','std','tau')) for t in clean}
    ref_u={}; ref_w={}
    for t in clean:
        others=[u for u in clean if u!=t]
        ref_u[t]=tuple(x.float().cuda() for x in ref_unif([patch[clean[u]['id']] for u in others]))
        ref_w[t]=tuple(x.float().cuda() for x in ref_weighted([fit_weighted(patch[clean[u]['id']],att[clean[u]['id']]) for u in others]))
    # assertions
    for t in clean:
        for nm,r in [('full',ref_full[t]),('u',ref_u[t]),('w',ref_w[t])]:
            assert float(r[1].min())>1e-3,(nm,t,float(r[1].min()))
        assert float((ref_w[t][0]-ref_u[t][0]).abs().max())>1e-2,t
    zs=[torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(s)) for s in (11,22,33,44)]
    clean_ref={t:[predict(raw[clean[t]['id']],z,'frozen',None) for z in zs] for t in clean}
    def active_rate(arm,t):
        p=patch[clean[t]['id']]; w=att[clean[t]['id']]
        if arm in ('v1_full','spatial_full'): ref=ref_full[t]
        elif arm in ('v1_t0','spatial_t0'): ref=ref_u[t]
        else: ref=ref_w[t]
        if arm=='weighted_t0':
            ww=wtil(w); m=(ww[:,:,None]*p).sum(1); sd=((ww[:,:,None]*(p-m[:,None,:]).square()).sum(1)+1e-6).sqrt()
        else:
            m=p.mean(1); sd=p.std(1,unbiased=False)
        return float((score(m,sd,ref[0],ref[1])>ref[2]).float().mean())
    def active_rate_case(rid,arm):
        p=patch[rid]; w=att[rid]
        if arm in ('v1_full','spatial_full'): ref=ref_full[rid.split('_')[0]] if False else ref_full[next(t for t in clean if clean[t]['id']==(rid if rid.startswith('clean_') else ''))] if False else None
        # resolve task by case id
        task=next(t for t in clean if clean[t]['id']==rid) if rid.startswith('clean_') else None
        if task is None:
            task=next(r['task'] for r in t0 if r['id']==rid)
        if arm in ('v1_full','spatial_full'): ref=ref_full[task]
        elif arm in ('v1_t0','spatial_t0'): ref=ref_u[task]
        else: ref=ref_w[task]
        if arm=='weighted_t0':
            ww=wtil(w); m=(ww[:,:,None]*p).sum(1); sd=((ww[:,:,None]*(p-m[:,None,:]).square()).sum(1)+1e-6).sqrt()
        else:
            m=p.mean(1); sd=p.std(1,unbiased=False)
        return float((score(m,sd,ref[0],ref[1])>ref[2]).float().mean())
    res=[]
    for r in t0:
        for zi,z in enumerate(zs):
            fz=predict(raw[r['id']],z,'frozen',None); bx,br=dist(fz,clean_ref[r['task']][zi])
            for arm in ARMS:
                ref=ref_full[r['task']] if arm.endswith('_full') else (ref_u[r['task']] if arm in ('v1_t0','spatial_t0') else ref_w[r['task']])
                a=predict(raw[r['id']],z,arm,ref,att[r['id']] if arm.startswith('spatial') or arm=='weighted_t0' else None); ax,ar=dist(a,clean_ref[r['task']][zi]); res.append({'id':r['id'],'task':r['task'],'factor':r['factor'],'sub':r['sub'],'arm':arm,'z':zi,'near_xyz':(bx-ax)/(bx+1e-12)*100 if r['factor']!='clean' else 0.0,'near_rot':(br-ar)/(br+1e-12)*100 if r['factor']!='clean' else 0.0,'drift_xyz':float((a-fz).abs()[:,:3].mean()),'drift_rot':float((a-fz).abs()[:,3:6].mean()),'active_rate':active_rate_case(r['id'],arm)})
    # geometry
    geom=[]
    caseids=sorted({r['case'] for r in moves if r.get('factor')=='clean'})
    for caseid in caseids:
        zr=next(r for r in moves if r['case']==caseid and np.allclose(r.get('offset_m',[9,9,9]),[0,0,0])); nz=[r for r in moves if r['case']==caseid and not np.allclose(r.get('offset_m',[9,9,9]),[0,0,0])]; task=zr['task']
        for r in nz:
            for zi,z in enumerate(zs):
                f0=predict(raw[zr['id']],z,'frozen',None); f1=predict(raw[r['id']],z,'frozen',None); df=(f1-f0).numpy()[:,:3].mean(0)
                for arm in ARMS:
                    ref=ref_full[task] if arm.endswith('_full') else (ref_u[task] if arm in ('v1_t0','spatial_t0') else ref_w[task]); a0=predict(raw[zr['id']],z,arm,ref,att[zr['id']] if arm.startswith('spatial') or arm=='weighted_t0' else None); a1=predict(raw[r['id']],z,arm,ref,att[r['id']] if arm.startswith('spatial') or arm=='weighted_t0' else None); da=(a1-a0).numpy()[:,:3].mean(0); cos=float(np.dot(da,df)/(np.linalg.norm(da)*np.linalg.norm(df)+1e-12)); ratio=float(np.linalg.norm(da)/(np.linalg.norm(df)+1e-12)); geom.append({'arm':arm,'case':task,'cos':cos,'ratio':ratio})
    groups={}
    factors=['clean','sensor_noise','background','lighting']
    for fac in factors:
        for arm in ARMS:
            sub=[x for x in res if x['factor']==fac and x['arm']==arm]; groups[f'{fac}__{arm}']={'n':len(sub),'near_xyz':float(np.mean([x['near_xyz'] for x in sub])),'near_rot':float(np.mean([x['near_rot'] for x in sub])),'drift_xyz':float(np.mean([x['drift_xyz'] for x in sub])),'drift_rot':float(np.mean([x['drift_rot'] for x in sub])),'active_rate':float(np.mean([x['active_rate'] for x in sub]))}
    for fac in ['table_texture','scene_texture']:
        for arm in ARMS:
            sub=[x for x in res if x['factor']=='background' and fac in x['sub'] and x['arm']==arm]; groups[f'background_{fac}__{arm}']={'n':len(sub),'near_xyz':float(np.mean([x['near_xyz'] for x in sub])),'near_rot':float(np.mean([x['near_rot'] for x in sub]))}
    for arm in ARMS:
        sub=[x for x in geom if x['arm']==arm]; groups[f'geom__{arm}']={'n':len(sub),'cos_median':float(np.median([x['cos'] for x in sub])),'ratio_median':float(np.median([x['ratio'] for x in sub]))}
    OUT.mkdir(exist_ok=True); (OUT/'actions_v3.json').write_text(json.dumps({'groups':groups,'rows':res,'geometry':geom},indent=2)+'\n'); print(json.dumps(groups,indent=2))
if __name__=='__main__': main()
