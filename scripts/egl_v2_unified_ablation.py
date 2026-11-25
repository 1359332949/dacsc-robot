"""Spatial Dev V2 unified shallow-layer ablation: 5 arms x 130 conditions x 3 seeds (EGL)."""
from __future__ import annotations
import argparse,ctypes,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
from dacsc.static_visual_v2 import array_hash,authorize,file_hash,seed_for,sensor_transform
from dacsc.single_block_probe import VIEWS,align,fit,moments,score
from egl_shallow_block3_closed_loop import observed,match_pixels
MANIFEST='d3e0d8c642cd215c879d2f0174663ff5ec585cd5c6d6b531a213c26e9fb982c0'
CHECKPOINT='f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6'
ARMS=('frozen','v1_b3_hard075','v2_prog_b3_b6','v2_soft_b3','v2_hard_matched_b3')
LAYERS=(3,6)

def refs_path(out,l): return out/f'full_l{l}.npz'
def refs_sha_path(out,l): return out/f'full_l{l}.sha256'
def load_refs(out,l):
    with np.load(refs_path(out,l),allow_pickle=False) as z:
        return {t:tuple(__import__('torch').as_tensor(z[f'{t}__{l}__{p}'],device='cuda:0') for p in ('mean','std','tau')) for t in {k.split('__')[0] for k in z.files}}
def fit_layer(source,l):
    raw={cid:np.load(p,allow_pickle=False)['b%d'%l] for cid,p in source.items()}
    for x in raw.values(): assert x.ndim==4 and x.shape[1:]==(2,2,384) and np.isfinite(x).all()
    refs={}; audit={}
    for t in sorted(raw):
        others=sorted(set(raw)-{t}); assert len(others)==9
        def stats(keys): return fit(__import__('torch').cat([__import__('torch').from_numpy(raw[k]) for k in keys]).float())
        mu,sd=stats(others); ss=[]
        for hold in others:
            inner=[k for k in others if k!=hold]; mi,si=stats(inner); m,q=__import__('torch').from_numpy(raw[hold]).float().unbind(1)
            ss.append(score(m,(q-m.square()).clamp_min(0).sqrt(),mi,si))
        tau=__import__('torch').cat(ss).quantile(.95,dim=0); audit[t]=dict(tasks=others,frames=sum(len(raw[k]) for k in others))
        for n,x in zip(('mean','std','tau'),(mu,sd,tau)): refs[f'{t}__{l}__{n}']=x.float().numpy()
    np.savez_compressed(refs_path(source['__out__'],l) if '__out__' in source else Path('/tmp/x.npz'),**refs)
    return refs,audit

def align_hard(x,ref,alpha):
    if alpha==0: return x
    mu,ss,tau=ref; p=x[:,1:]; m=p.mean(1); sd=p.std(1,unbiased=False)
    active=score(m,sd,mu,ss)>tau
    gain=(((1-alpha)*sd+alpha*ss)/(sd.square()+1e-6).sqrt()).clamp(.5,2)
    out=__import__('torch').where(active[:,None,:],gain[:,None,:]*(p-m[:,None,:])+((1-alpha)*m+alpha*mu)[:,None,:],p)
    return __import__('torch').cat((x[:,:1],out),dim=1)

def alpha_soft(patch,mu,ss,tau,cap=.75):
    m=patch.mean(1); sd=patch.std(1,unbiased=False); d=score(m,sd,mu,ss)
    a=cap*((d-tau).clamp_min(0)/(d+1e-6)); return __import__('torch').where(d>tau,a,__import__('torch').zeros_like(a))
def align_soft(x,ref,cap=.75):
    mu,ss,tau=ref; p=x[:,1:]; a=alpha_soft(p,mu,ss,tau,cap)
    m=p.mean(1); sd=p.std(1,unbiased=False); ab=a[:,None,:]; mb=m[:,None,:]; mub=mu[:,None,:]; ssb=ss[:,None,:]
    gain=(((1-ab)*sd[:,None,:]+ab*ssb)/(sd[:,None,:].square()+1e-6).sqrt()).clamp(.5,2)
    active=ab>0
    out=__import__('torch').where(active,gain*(p-mb)+(1-ab)*mb+ab*mub,p)
    return __import__('torch').cat((x[:,:1],out),dim=1)

def main():
    import torch
    from corp import load_policy
    from target_only_action_validation import sample_z,visual_from_obs
    from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.benchmark import grab_language_from_filename
    import libero.libero.envs.env_wrapper as wrapper
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('source','fit','preflight','run')); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--workers',type=int,default=1); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--matched-alpha',type=float,default=None); ap.add_argument('--pilot',action='store_true'); a=ap.parse_args()
    assert os.environ['MUJOCO_GL']==os.environ['PYOPENGL_PLATFORM']=='egl'
    final=ROOT/'docs/data/static_visual_v2_suite/final'; mp=final/'dev_manifest.json'; assert file_hash(mp)==MANIFEST
    m=json.loads(mp.read_text()); base=sorted(m['cases'],key=lambda c:c['case_id']); assert len(base)==100
    for c in base: authorize(m,c,'V2 EGL ablation',MANIFEST,mp)
    clean={c['base_task']:c['case_id'] for c in base if c['factor']=='clean'}; assert len(clean)==10
    cp=CORP/'checkpoints/corp_libero.pt'; assert file_hash(cp)==CHECKPOINT
    policy=load_policy(cp,device='cuda:0').eval().requires_grad_(False); ext=policy.representation_extractor; assert tuple(ext.view_names)==VIEWS and len(ext.visual_encoder.backbone.blocks)==12
    cases=base+[dict(c, _dual=True) for c in base if c['factor']=='sensor_noise']; assert len(cases)==130
    def stream_seeds(c,s):
        r=json.loads((final/'results'/(c['case_id']+'.json')).read_text())['seeds']; es=r['environment_seed']; ns=r['sensor_noise_seed']; ps=r['policy_flow_seed'] if s==0 else seed_for(c['pair_group_id'],s,'egl_policy_seed'); return es,ns,ps
    def initial_of(c): return torch.load(LIB/c['init_state_file'],weights_only=False,map_location='cpu')[c['init_state_index']]
    def lang_of(c): return grab_language_from_filename(c['suite'],c['base_task']+'.bddl')
    def run_arm(c,s,arm,refs,matched):
        es,ns,ps=stream_seeds(c,s); code=int(c['runtime_params']['noise']); dual=bool(c.get('_dual',False)); init=initial_of(c); lang=lang_of(c); random.seed(es); np.random.seed(es); torch.manual_seed(es); ctypes.CDLL(None).srand(ctypes.c_uint(es))
        env=OffScreenRenderEnv(bddl_file_name=str(LIB/c['bddl']),camera_heights=256,camera_widths=256,render_gpu_device_id=0); hooks=[]
        try:
            env.noise=0; env.seed(es); env.reset(); obs=env.set_init_state(init)
            for _ in range(SETTLING_STEPS): obs=env.step(DUMMY_ACTION)[0]
            task=c['base_task']; b3,b6=refs[3][task],refs[6][task]
            if arm=='v1_b3_hard075': hooks.append(ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda mod,args,x:align_hard(x,b3,.75)))
            elif arm=='v2_prog_b3_b6':
                hooks.append(ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda mod,args,x:align_hard(x,b3,.5)))
                hooks.append(ext.visual_encoder.backbone.blocks[5].register_forward_hook(lambda mod,args,x:align_hard(x,b6,.5)))
            elif arm=='v2_soft_b3': hooks.append(ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda mod,args,x:align_soft(x,b3,.75)))
            elif arm=='v2_hard_matched_b3': hooks.append(ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda mod,args,x:align_hard(x,b3,float(matched))))
            g=torch.Generator(device='cuda:0').manual_seed(ps); steps=0; success=False; cache={}; audit=[]
            with torch.no_grad():
                while steps<220:
                    raw=observed(obs,code,ns,steps,dual,wrapper); key=(steps,array_hash(env.sim.get_state().flatten()),*(array_hash(np.asarray(raw[k])) for k in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'))); raw=match_pixels(raw,key,cache,audit); po=policy_observation(raw,lang,policy); z=torch.randn((1,12,7),device='cuda:0',generator=g); _,rep=visual_from_obs(policy,po); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu().numpy()
                    for action in plan:
                        obs,rew,done,info=env.step(action); steps+=1; success=bool(is_success(env,info,rew,done))
                        if done or success or steps>=220: break
                    if done or success: break
            return {'arm':arm,'success':success,'steps':steps,'dual':dual,'seed':s,'case_id':c['case_id'],'factor':c['factor'],'sub':c.get('sub_dimension')}
        finally:
            for h in hooks: h.remove()
            env.close()
    if a.mode=='source':
        cases=[c for c in base if c['factor']=='clean']; assert len(cases)==10
        (a.out/'source').mkdir(parents=True,exist_ok=True); subs={}
        for c in cases[a.shard::a.workers]:
            es,ns,ps=stream_seeds(c,0); init=initial_of(c); lang=lang_of(c); random.seed(es); np.random.seed(es); torch.manual_seed(es); ctypes.CDLL(None).srand(ctypes.c_uint(es)); env=OffScreenRenderEnv(bddl_file_name=str(LIB/c['bddl']),camera_heights=256,camera_widths=256,render_gpu_device_id=0); captured={3:[],6:[]}
            def cap(l):
                def fn(mod,args,x): captured[l].append(np.stack([m.detach().double().cpu().numpy() for m in moments(x[:,1:].double())])); return None
                return fn
            hooks=[ext.visual_encoder.backbone.blocks[l-1].register_forward_hook(cap(l)) for l in LAYERS]
            try:
                env.noise=0; env.seed(es); env.reset(); obs=env.set_init_state(init)
                for _ in range(SETTLING_STEPS): obs=env.step(DUMMY_ACTION)[0]
                g=torch.Generator(device='cuda:0').manual_seed(ps); steps=0
                with torch.no_grad():
                    while steps<220:
                        raw=sensor_transform(obs,int(c['runtime_params']['noise']),ns,steps,wrapper) if c['factor']=='sensor_noise' else obs; po=policy_observation(raw,lang,policy); z=torch.randn((1,12,7),device='cuda:0',generator=g); _,rep=visual_from_obs(policy,po); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu().numpy()
                        for action in plan:
                            obs,rew,done,info=env.step(action); steps+=1; success=bool(is_success(env,info,rew,done))
                            if done or success or steps>=220: break
                        if done or success: break
                assert len(captured[3])==len(captured[6])
                np.savez_compressed(a.out/'source'/(c['case_id']+'.npz'),b3=np.stack(captured[3]),b6=np.stack(captured[6]))
                print(c['case_id'],len(captured[3]),flush=True)
            finally:
                for h in hooks: h.remove()
                env.close()
        return
    if a.mode=='fit':
        source={c['base_task']:a.out/'source'/(c['case_id']+'.npz') for c in base if c['factor']=='clean'}; assert all(p.exists() for p in source.values())
        for l in LAYERS:
            raw={cid:np.load(p,allow_pickle=False)['b%d'%l] for cid,p in source.items()}; refs2={}; audit={}
            for t in sorted(raw):
                others=sorted(set(raw)-{t})
                def stats(keys): return fit(torch.cat([torch.from_numpy(raw[k]) for k in keys]).float())
                mu,sd=stats(others); ss=[]
                for hold in others:
                    inner=[k for k in others if k!=hold]; mi,si=stats(inner); mm,qq=torch.from_numpy(raw[hold]).float().unbind(1); ss.append(score(mm,(qq-mm.square()).clamp_min(0).sqrt(),mi,si))
                tau=torch.cat(ss).quantile(.95,dim=0); audit[t]=dict(tasks=others,frames=sum(len(raw[k]) for k in others))
                for n,x in zip(('mean','std','tau'),(mu,sd,tau)): refs2[f'{t}__{l}__{n}']=x.float().numpy()
            np.savez_compressed(refs_path(a.out,l),**refs2); refs_sha_path(a.out,l).write_text(file_hash(refs_path(a.out,l))); (a.out/f'folds_l{l}.json').write_text(json.dumps(audit,indent=2))
        return
    refs={l:load_refs(a.out,l) for l in LAYERS}
    if a.mode=='preflight':
        vals=[]; tasks={3:refs[3],6:refs[6]}
        for c in cases[a.shard::a.workers]:
            es,ns,ps=stream_seeds(c,0); init=initial_of(c); lang=lang_of(c); random.seed(es); np.random.seed(es); torch.manual_seed(es); ctypes.CDLL(None).srand(ctypes.c_uint(es)); env=OffScreenRenderEnv(bddl_file_name=str(LIB/c['bddl']),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
            def fn(mod,args,x):
                mu,ss,tau=tasks[3][c['base_task']]; p=x[:,1:]; a=alpha_soft(p,mu,ss,tau,.75); vals.extend(a[a>0].flatten().tolist()); return None
            h=ext.visual_encoder.backbone.blocks[2].register_forward_hook(fn)
            try:
                env.noise=0; env.seed(es); env.reset(); obs=env.set_init_state(init)
                for _ in range(SETTLING_STEPS): obs=env.step(DUMMY_ACTION)[0]
                raw=observed(obs,int(c['runtime_params']['noise']),ns,0,bool(c.get('_dual',False)),wrapper); po=policy_observation(raw,lang,policy); z=torch.zeros((1,12,7),device='cuda:0'); _,rep=visual_from_obs(policy,po); _=sample_z(policy,rep,z)
            finally:
                h.remove(); env.close()
        Path(a.out/f'preflight_{a.shard}.json').write_text(json.dumps(vals))
        return
    matched=float(a.matched_alpha) if a.matched_alpha is not None else None; assert matched is not None
    # run: dynamic claim-free sharding over 390 conditions; fixed arm order
    jobs=[(c,s) for s in (0,1,2) for c in cases]
    if a.pilot: jobs=jobs[:2]
    (a.out/'trials').mkdir(parents=True,exist_ok=True)
    for c,s in jobs[a.shard::a.workers]:
        regime='dual' if c.get('_dual') else 'standard'
        dest=a.out/'trials'/f'{regime}_{c["case_id"]}_{s}.json'
        if dest.exists(): continue
        rows=[run_arm(c,s,arm,refs,matched) for arm in ARMS]
        rows[0]['env_seed'],rows[0]['sensor_seed'],rows[0]['policy_seed']=stream_seeds(c,s)
        tmp=dest.with_suffix('.pending'); tmp.write_text(json.dumps(rows,allow_nan=False)); tmp.rename(dest); print(dest.name,flush=True)
if __name__=='__main__': main()
