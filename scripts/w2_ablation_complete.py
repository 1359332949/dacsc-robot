"""W2-shrinkage zero-hyperparameter probe on cached Spatial t=0 and bowl moves."""
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
INPUT=ROOT/'runs/invariance_sensitivity_v1b/input'; OUT=ROOT/'runs/w2_probe_v1'
ARMS=('v1_hard075','v1_hard075_rms','w2_proj_q95','js_shrink_q95','w2_proj_rms','w2_proj_rms_noclamp','js_shrink_rms','js_shrink_rms_noclamp')
policy=None

def raw_row(r):
    z=np.load(INPUT/(r['id']+'.npz'),allow_pickle=False); return {'images':{v:torch.from_numpy(z[v].copy())[None] for v in ('agentview','eye_in_hand')},'state':torch.from_numpy(z['state'].copy()),'instruction':[r['lang']]}
def prep(raw): return prepare_observation(raw,config=policy.config,normalizer=policy.normalizer,device='cuda:0')
def alpha_map(arm,d2,tau,rq,r_rms):
    d=torch.sqrt(d2.clamp_min(0))+1e-8
    if arm=='v1_hard075': return torch.where(d2>tau,torch.full_like(d2,.75),torch.zeros_like(d2))
    if arm=='v1_hard075_rms': return torch.where(d>r_rms,torch.full_like(d2,.75),torch.zeros_like(d2))
    if arm=='w2_proj_q95': return (1-rq/d).clamp(0,1)
    if arm=='js_shrink_q95': return (1-tau/(d2+1e-8)).clamp(0,1)
    if arm in ('w2_proj_rms','w2_proj_rms_noclamp'): return (1-r_rms/d).clamp(0,1)
    if arm in ('js_shrink_rms','js_shrink_rms_noclamp'): return (1-r_rms*r_rms/(d2+1e-8)).clamp(0,1)
def apply_arm(x,ref,arm,tau,rq,r_rms):
    mu,ss,_=ref; p=x[:,1:]; m=p.mean(1); sd=p.std(1,unbiased=False); d2=score(m,sd,mu,ss); a=alpha_map(arm,d2,tau,rq,r_rms); a3=a[:,None,:]; gain=(((1-a3)*sd[:,None,:]+a3*ss[:,None,:])/(sd[:,None,:].square()+1e-6).sqrt())
    if arm not in ('js_shrink_rms_noclamp','w2_proj_rms_noclamp'): gain=gain.clamp(.5,2)
    out=torch.where(a3>0,gain*(p-m[:,None,:])+((1-a3)*m[:,None,:]+a3*mu[:,None,:]),p); return torch.cat((x[:,:1],out),1),a,gain
def rms_radius(clean,source):
    rr={}
    for t in clean:
        others=[u for u in clean if u!=t]; vals=[]
        for hold in others:
            inner=[u for u in others if u!=hold]; ms=[]; qs=[]
            for u in inner:
                z=torch.from_numpy(source[clean[u]['id']]).double(); m,q=z.unbind(1)  # [T,2,384]
                ms.append(m.reshape(-1,2,384)); qs.append(q.reshape(-1,2,384))
            M=torch.cat(ms,0); Q=torch.cat(qs,0); mi=M.mean(0); si=(Q.mean(0)-mi.square()).clamp_min(0).sqrt().clamp_min(1e-3)
            z=torch.from_numpy(source[clean[hold]['id']]).double(); m,q=z.unbind(1); m=m.reshape(-1,2,384); q=q.reshape(-1,2,384)
            vals.append(score(m,(q-m.square()).clamp_min(0).sqrt(),mi,si))  # [Th,2,384]
        r=torch.cat(vals,dim=0).mean(0).sqrt().float().cuda(); assert r.shape==(2,384) and torch.isfinite(r).all() and float(r.max()/r.min())>1.5,(t,r.shape,float(r.max()/r.min())); rr[t]=r
    return rr

def predict(raw,z,arm,fullref,ref,tau,rq,r_rms):
    h=None; holder={}
    def hook(mod,args,x):
        out,a,g=apply_arm(x,ref,arm,tau,rq,r_rms); holder['a']=a.detach(); holder['g']=g.detach(); return out
    if arm!='frozen': h=policy.representation_extractor.visual_encoder.backbone.blocks[2].register_forward_hook(hook)
    try:
        _,rep=visual_from_obs(policy,raw); out=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu(); return out,holder
    finally:
        if h: h.remove()
def dist(a,b):
    d=(a-b).abs(); return float(d[:,:3].mean()),float(d[:,3:6].mean())
def main():
    global policy
    policy=load_policy(CORP/'checkpoints/corP_libero.pt' if False else CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False)
    rows=[]
    for f in sorted(INPUT.glob('index_*.json')): rows+=json.load(open(f))
    t0=[r for r in rows if not r.get('geometry')]; moves=[r for r in rows if r.get('geometry')]
    raw={r['id']:raw_row(r) for r in rows}
    clean={r['task']:r for r in t0 if r['factor']=='clean'}; assert len(clean)==10
    full=np.load(ROOT/'runs/full_traj_v1/full.npz',allow_pickle=False); ref={t:tuple(torch.as_tensor(full[f'{t}__3__{k}'],device='cuda:0').float() for k in ('mean','std','tau')) for t in clean}
    source={f.stem:np.load(f) for f in (ROOT/'runs/full_traj_v1/source').glob('*.npy')}; rr=rms_radius(clean,source)
    rq={t:torch.sqrt(ref[t][2].clamp_min(0)) for t in clean}; tau={t:ref[t][2] for t in clean}
    # clean frozen refs
    zs=[torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(s)) for s in (11,22,33,44)]
    clean_ref={t:[predict(raw[clean[t]['id']],z,'frozen',ref[t],ref[t],tau[t],rq[t],rr[t])[0] for z in zs] for t in clean}
    res=[]
    for r in t0:
        for zi,z in enumerate(zs):
            fz=predict(raw[r['id']],z,'frozen',ref[r['task']],ref[r['task']],tau[r['task']],rq[r['task']],rr[r['task']])[0]; bx,br=dist(fz,clean_ref[r['task']][zi])
            for arm in ARMS:
                a,holder=predict(raw[r['id']],z,arm,ref[r['task']],ref[r['task']],tau[r['task']],rq[r['task']],rr[r['task']]); ax,ar=dist(a,clean_ref[r['task']][zi]); al=holder.get('a'); gm=holder.get('g')
                res.append({'id':r['id'],'task':r['task'],'factor':r['factor'],'sub':r['sub'],'arm':arm,'z':zi,'near_xyz':(bx-ax)/(bx+1e-12)*100 if r['factor']!='clean' else 0.0,'near_rot':(br-ar)/(br+1e-12)*100 if r['factor']!='clean' else 0.0,'drift_xyz':float((a-fz).abs()[:,:3].mean()),'drift_rot':float((a-fz).abs()[:,3:6].mean()),'mean_alpha':float(al.mean()) if al is not None else 0.0,'active_rate':float((al>0).float().mean()) if al is not None else 0.0,'clamp_frac':float(((gm<=.5000001)|(gm>=1.999999)).float().mean()) if gm is not None else 0.0})
    # geometry
    geom=[]; caseids=sorted({r['case'] for r in moves if r.get('factor')=='clean'})
    for caseid in caseids:
        zr=next(r for r in moves if r['case']==caseid and np.allclose(r.get('offset_m',[9,9,9]),[0,0,0])); nz=[r for r in moves if r['case']==caseid and not np.allclose(r.get('offset_m',[9,9,9]),[0,0,0])]; task=zr['task']
        for r in nz:
            for zi,z in enumerate(zs):
                f0,_=predict(raw[zr['id']],z,'frozen',ref[task],ref[task],tau[task],rq[task],rr[task]); f1,_=predict(raw[r['id']],z,'frozen',ref[task],ref[task],tau[task],rq[task],rr[task]); df=(f1-f0).numpy()[:,:3].mean(0)
                for arm in ARMS:
                    a0,_=predict(raw[zr['id']],z,arm,ref[task],ref[task],tau[task],rq[task],rr[task]); a1,_=predict(raw[r['id']],z,arm,ref[task],ref[task],tau[task],rq[task],rr[task]); da=(a1-a0).numpy()[:,:3].mean(0); cos=float(np.dot(da,df)/(np.linalg.norm(da)*np.linalg.norm(df)+1e-12)); ratio=float(np.linalg.norm(da)/(np.linalg.norm(df)+1e-12)); geom.append({'arm':arm,'case':task,'cos':cos,'ratio':ratio})
    groups={}
    for fac in ['clean','sensor_noise','background','lighting']:
        for arm in ARMS:
            sub=[x for x in res if x['factor']==fac and x['arm']==arm]; groups[f'{fac}__{arm}']={'n':len(sub),'near_xyz':float(np.mean([x['near_xyz'] for x in sub])),'near_rot':float(np.mean([x['near_rot'] for x in sub])),'drift_xyz':float(np.mean([x['drift_xyz'] for x in sub])),'drift_rot':float(np.mean([x['drift_rot'] for x in sub])),'mean_alpha':float(np.mean([x['mean_alpha'] for x in sub])),'active_rate':float(np.mean([x['active_rate'] for x in sub])),'clamp_frac':float(np.mean([x['clamp_frac'] for x in sub]))}
    for fac in ['table_texture','scene_texture']:
        for arm in ARMS:
            sub=[x for x in res if x['factor']=='background' and fac in x['sub'] and x['arm']==arm]; groups[f'background_{fac}__{arm}']={'n':len(sub),'near_xyz':float(np.mean([x['near_xyz'] for x in sub])),'near_rot':float(np.mean([x['near_rot'] for x in sub])),'mean_alpha':float(np.mean([x['mean_alpha'] for x in sub]))}
    for arm in ARMS:
        sub=[x for x in geom if x['arm']==arm]; groups[f'geom__{arm}']={'n':len(sub),'cos_median':float(np.median([x['cos'] for x in sub])),'ratio_median':float(np.median([x['ratio'] for x in sub]))}
    OUT.mkdir(exist_ok=True); (OUT/'probe.json').write_text(json.dumps({'r_rms':{t:rr[t].detach().cpu().tolist() for t in rr},'groups':groups,'rows':res,'geometry':geom},indent=2)+'\n')
    lines=['# W2 complete 8-arm ablation','', '| Group | '+' | '.join(ARMS)+' |','|---|'+'---:|'*len(ARMS)]
    for fac in ['clean','sensor_noise','background','lighting']:
        for metric in ['near_xyz','near_rot']:
            lines.append('| '+fac+' '+metric+' | '+' | '.join(f"{groups[f'{fac}__{a}'][metric]:.2f}" for a in ARMS)+' |')
    lines+=['','## Deltas vs v1_hard075','', '| Group/metric | '+' | '.join(ARMS[1:])+' |','|---|'+'---:|'*(len(ARMS)-1)]
    for fac in ['clean','sensor_noise','background','lighting']:
        for metric in ['near_xyz','near_rot']:
            base=groups[f'{fac}__v1_hard075'][metric]
            lines.append('| '+fac+' '+metric+' | '+' | '.join(f"{groups[f'{fac}__{a}'][metric]-base:+.2f}" for a in ARMS[1:])+' |')
    (OUT/'ablation.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines)); print(json.dumps(groups,indent=2))
if __name__=='__main__': main()
