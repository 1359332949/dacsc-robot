"""EAGA action probe on cached t=0 Dev and physical bowl moves (no closed loop)."""
from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
from corp import load_policy
from corp.preprocessing import prepare_observation
from dacsc.single_block_probe import align,fit,moments,score
from target_only_action_validation import sample_z,visual_from_obs
INPUT=ROOT/'runs/invariance_sensitivity_v1b/input'; OUT=ROOT/'runs/eaga_probe_v1'
policy=None

def load_row(r):
    z=np.load(INPUT/(r['id']+'.npz'),allow_pickle=False); raw={'images':{v:torch.from_numpy(z[v].copy())[None] for v in ('agentview','eye_in_hand')},'state':torch.from_numpy(z['state'].copy()),'instruction':[r['lang']]}; return prepare_observation(raw,config=policy.config,normalizer=policy.normalizer,device='cuda:0')
def attention(po):
    ex=policy.representation_extractor; rs=ex.cross_attention_resampler; p=ex.visual_encoder(torch.cat([po['images'][v] for v in ex.view_names])); B=po['state'].shape[0]; ws=[]
    for i,v in enumerate(ex.view_names):
        pt=p[i*B:(i+1)*B]; lang,pool=ex.language_encoder(po['instruction']); gamma,beta=rs.film(pool).unsqueeze(1).chunk(2,dim=-1); q=rs.queries[i].unsqueeze(0).expand(B,-1,-1); q=q*(1+gamma)+beta; patches=rs.input_projection(pt)+rs.view_embedding[i].unsqueeze(0)
        _,w=rs.cross_attention(q,patches,patches,need_weights=True,average_attn_weights=False); ws.append(w.mean(1).mean(1)[0])
    return torch.stack(ws)
def predict(po,z):
    _,rep=visual_from_obs(policy,po); return policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu()
def align_weighted(x,ref,alpha=.75):
    mu,ss,tau=ref; p=x[:,1:]; m=p.mean(1); sd=p.std(1,unbiased=False); active=score(m,sd,mu,ss)>tau; gain=(((1-alpha)*sd+alpha*ss)/(sd.square()+1e-6).sqrt()).clamp(.5,2); out=torch.where(active[:,None,:],gain[:,None,:]*(p-m[:,None,:])+((1-alpha)*m+alpha*mu)[:,None,:],p); return torch.cat((x[:,:1],out),1)
def align_spatial(x,ref,w,pref=.35,bg=.75):
    mu,ss,tau=ref; p=x[:,1:]; m=p.mean(1); sd=p.std(1,unbiased=False); active=score(m,sd,mu,ss)>tau
    wn=(w/w.amax(-1,keepdim=True)).clamp(0,1); a=(pref+(bg-pref)*(1-wn)).clamp(pref,bg); base=(((1-.75)*sd+.75*ss)/(sd.square()+1e-6).sqrt()).clamp(.5,2)
    # blend per-patch alpha into gain; inactive bypass
    a3=a[:,:,None]; g=a3*base[:,None,:]+(1-a3)*1.0; out=torch.where(active[:,None,:],g*(p-m[:,None,:])+((1-a3)*m[:,None,:]+a3*mu[:,None,:]),p); return torch.cat((x[:,:1],out),1)
def main():
    global policy
    policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False)
    rows=[]
    for f in sorted(INPUT.glob('index_*.json')): rows+=json.load(open(f))
    t0=[r for r in rows if not r.get('geometry')]; moves=[r for r in rows if r.get('geometry')]
    full=np.load(ROOT/'runs/full_traj_v1/full.npz',allow_pickle=False); uniform={}
    for t in {r['task'] for r in t0}: uniform[t]=tuple(torch.as_tensor(full[f'{t}__3__{k}'],device='cuda:0') for k in ('mean','std','tau'))
    # source weighted moments from full-trajectory source sketches; clean t0 attention replicated across frames
    source={}
    for f in (ROOT/'runs/full_traj_v1/source').glob('*.npy'):
        z=np.load(f); source[f.stem]=z
    cleanrows={r['task']:r for r in t0 if r['factor']=='clean'}; assert len(cleanrows)==10
    clean_w={}
    for t,r in cleanrows.items(): clean_w[t]=attention(load_row(r))
    weighted={}
    for t in uniform:
        others=[u for u in uniform if u!=t]; ms=[]; qs=[]; ws=[]
        for u in others:
            cid=cleanrows[u]['id']; arr=torch.from_numpy(source[cid]).float()  # [F,2,2,384]
            w=clean_w[u].double(); w=.8*w+.2/256; w=w/w.sum(-1,keepdim=True); ws.append(w); ms.append(arr[:,:,0,:]); qs.append(arr[:,:,1,:])
        M=torch.cat(ms,0); Q=torch.cat(qs,0); W=torch.stack(ws,0).mean(0).cpu()
        Mm=M.mean(0); Qm=Q.mean(0); wsum=W.sum(1)
        mu=(Mm[:,None,:]*W[:,:,None]).sum(1)/wsum[:,None]; var=((Qm-Mm.square())[:,None,:]*W[:,:,None]).sum(1)/wsum[:,None]; weighted[t]=(mu.float().cuda(),var.clamp_min(0).sqrt().float().cuda(),uniform[t][2].float())
    # t0 action comparison
    def action(po,z,arm,ref,w=None):
        h=None
        if arm=='v1': h=policy.representation_extractor.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align(x,ref,.75,'dead'))
        elif arm=='weighted': h=policy.representation_extractor.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align_weighted(x,ref,.75))
        elif arm=='spatial': h=policy.representation_extractor.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align_spatial(x,ref,w))
        try: return predict(po,z)
        finally:
            if h: h.remove()
    res=[]
    for r in t0:
        po=load_row(r); w=attention(po); z=torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(123)); base=predict(po,z)
        for arm in ('v1','weighted','spatial'):
            ref=uniform[r['task']] if arm=='v1' else weighted[r['task']]; a=action(po,z,arm,ref,w if arm=='spatial' else None); d=(a-base).abs(); res.append({'id':r['id'],'task':r['task'],'factor':r['factor'],'sub':r['sub'],'arm':arm,'xyz':float(d[:,:3].mean()),'rot':float(d[:,3:6].mean()),'all':float(d.mean())})
    # move geometry: direction cosine/ratio of action delta to physical offset, per arm
    byid={r['id']:r for r in moves}; geom=[]
    # zero move and nonzero move pairs by case
    for task in sorted({r['case'] for r in moves if r.get('factor')=='clean'}):
        zr=next((r for r in moves if r['case']==task and np.allclose(r.get('offset_m',[9,9,9]),[0,0,0])),None)
        if not zr: continue
        base_task=zr['task']
        po0=load_row(zr); z=torch.randn((1,12,7),device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(123)); w0=attention(po0)
        for r in [x for x in moves if x['case']==task and not np.allclose(x.get('offset_m',[9,9,9]),[0,0,0])]:
            po=load_row(r); w=attention(po); off=np.asarray(r['offset_m']);
            for arm in ('frozen','v1','weighted','spatial'):
                if arm=='frozen': a0=predict(po0,z); a1=predict(po,z)
                else:
                    ref=uniform[base_task] if arm=='v1' else weighted[base_task]; a0=action(po0,z,arm,ref,w0 if arm=='spatial' else None); a1=action(po,z,arm,ref,w if arm=='spatial' else None)
                d=(a1-a0).numpy()[:,:3].mean(0); cos=float(np.dot(d,off)/(np.linalg.norm(d)*np.linalg.norm(off)+1e-12)); ratio=float(np.linalg.norm(d)/(np.linalg.norm(off)+1e-12)); geom.append({'arm':arm,'case':base_task,'offset':off.tolist(),'cos':cos,'ratio':ratio})
    groups={}
    for fac in ['clean','sensor_noise','background','lighting']:
        for arm in ('v1','weighted','spatial'):
            sub=[x for x in res if x['factor']==fac and x['arm']==arm]
            groups[f'{fac}__{arm}']={'n':len(sub),'xyz':float(np.mean([x['xyz'] for x in sub])),'rot':float(np.mean([x['rot'] for x in sub]))}
    for arm in ('frozen','v1','weighted','spatial'):
        sub=[x for x in geom if x['arm']==arm]; groups[f'geom__{arm}']={'n':len(sub),'cos_median':float(np.median([x['cos'] for x in sub])),'ratio_median':float(np.median([x['ratio'] for x in sub])),'cos_pos':int(sum(x['cos']>0 for x in sub))}
    OUT.mkdir(exist_ok=True); (OUT/'actions.json').write_text(json.dumps({'groups':groups,'rows':res,'geometry':geom},indent=2)+'\n'); print(json.dumps(groups,indent=2))
if __name__=='__main__': main()
