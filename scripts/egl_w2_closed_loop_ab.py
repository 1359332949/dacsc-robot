"""4-arm EGL closed-loop A/B: frozen, V1, two zero-hyperparameter W2 arms."""
from __future__ import annotations
import argparse,json,sys,os,ctypes,random
from pathlib import Path
import numpy as np,torch
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'
from corp import load_policy
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation
from target_only_action_validation import sample_z,visual_from_obs
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import grab_language_from_filename
from dacsc.static_visual_v2 import array_hash,seed_for,sensor_transform
from dacsc.single_block_probe import align,score
from egl_shallow_block3_closed_loop import observed,match_pixels
import libero.libero.envs.env_wrapper as wrapper
ARMS=('frozen','v1_hard075','w2_proj_rms_noclamp','js_shrink_rms_noclamp')

def raw_row(r):
    z=np.load(INPUT/(r['id']+'.npz'),allow_pickle=False); return {'images':{v:torch.from_numpy(z[v].copy())[None] for v in ('agentview','eye_in_hand')},'state':torch.from_numpy(z['state'].copy()),'instruction':[r['lang']]}

def w2_apply(x,ref,arm,tau,rq,r_rms):
    mu,ss,_=ref; p=x[:,1:]; m=p.mean(1); sd=p.std(1,unbiased=False); d2=score(m,sd,mu,ss); d=torch.sqrt(d2.clamp_min(0))+1e-8
    if arm=='v1_hard075': a=torch.where(d2>tau,torch.full_like(d2,.75),torch.zeros_like(d2)); clamp=True
    elif arm=='w2_proj_rms_noclamp': a=(1-r_rms/d).clamp(0,1); clamp=False
    elif arm=='js_shrink_rms_noclamp': a=(1-r_rms*r_rms/(d2+1e-8)).clamp(0,1); clamp=False
    else: raise ValueError(arm)
    a3=a[:,None,:]; gain=(((1-a3)*sd[:,None,:]+a3*ss[:,None,:])/(sd[:,None,:].square()+1e-6).sqrt())
    if clamp: gain=gain.clamp(.5,2)
    out=torch.where(a3>0,gain*(p-m[:,None,:])+((1-a3)*m[:,None,:]+a3*mu[:,None,:]),p); out=torch.cat((x[:,:1],out),1); assert torch.isfinite(out).all(), arm
    return out

def rms_radius(clean,source):
    rr={}
    for t in clean:
        others=[u for u in clean if u!=t]; vals=[]
        for hold in others:
            inner=[u for u in others if u!=hold]; ms=[]; qs=[]
            for u in inner:
                z=torch.from_numpy(source[clean[u]]).double(); m,q=z.unbind(1); ms.append(m.reshape(-1,2,384)); qs.append(q.reshape(-1,2,384))
            mi=torch.cat(ms,0).mean(0); si=(torch.cat(qs,0).mean(0)-mi.square()).clamp_min(0).sqrt().clamp_min(1e-3); z=torch.from_numpy(source[clean[hold]]).double(); m,q=z.unbind(1); m=m.reshape(-1,2,384); q=q.reshape(-1,2,384); vals.append(score(m,(q-m.square()).clamp_min(0).sqrt(),mi,si))
        r=torch.cat(vals,dim=0).mean(0).sqrt().float().cuda(); assert r.shape==(2,384) and torch.isfinite(r).all() and float(r.max()/r.min())>1.5; rr[t]=r
    return rr

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--workers',type=int,default=16); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--max-steps',type=int,default=220); ap.add_argument('--seeds',default='0'); a=ap.parse_args(); seeds_list=[int(x) for x in a.seeds.split(',')]
    policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False); ext=policy.representation_extractor
    man=json.load(open(ROOT/'docs/data/static_visual_v2_suite/final/dev_manifest.json')); allc={c['case_id']:c for c in man['cases']}
    res={c['case_id']:json.load(open(ROOT/'docs/data/static_visual_v2_suite/final/results'/(c['case_id']+'.json'))) for c in man['cases']}
    conditions=[]
    for c in man['cases']:
        if c['factor']=='clean': conditions.append((c,False,'clean'))
        elif c['factor']=='sensor_noise': conditions.append((c,True,'sensor_bilateral'))
        elif c['factor']=='background': conditions.append((c,False,'background_'+('table_texture' if 'table' in c['sub_dimension'] else 'scene_texture')))
        elif c['factor']=='lighting': conditions.append((c,False,'lighting'))
    assert len(conditions)==100
    clean={c['task'] if 'task' in c else c['base_task']:c['case_id'] for c in man['cases'] if c['factor']=='clean'}
    # manifest uses base_task
    clean={c['base_task']:c['case_id'] for c in man['cases'] if c['factor']=='clean'}
    source={f.stem:np.load(f) for f in (ROOT/'runs/full_traj_v1/source').glob('*.npy')}; rr={}
    for c,fac,g in conditions:
        if c['base_task'] not in rr: rr[c['base_task']]=None
    # compute r_rms once for all tasks
    full=np.load(ROOT/'runs/full_traj_v1/full.npz',allow_pickle=False); refs={t:tuple(torch.as_tensor(full[f'{t}__3__{k}'],device='cuda:0').float() for k in ('mean','std','tau')) for t in clean}
    rr=rms_radius(clean,source)
    Path(a.out).mkdir(parents=True,exist_ok=True); (a.out/'trials').mkdir(exist_ok=True)
    jobs=[]
    for s in seeds_list:
        for c,dual,g in conditions: jobs.append((c,dual,g,s))
    for c,dual,g,s in jobs[a.shard::a.workers]:
        dest=a.out/'trials'/f'{g}_{c["case_id"]}_{s}.json'
        if dest.exists(): continue
        seeds=res[c['case_id']]['seeds']; es=seeds['environment_seed']; ns=seeds['sensor_noise_seed']; ps=seeds['policy_flow_seed'] if s==0 else seed_for(c['pair_group_id'],s,'egl_policy_seed'); init=torch.load(LIB/c['init_state_file'],weights_only=False,map_location='cpu')[c['init_state_index']]; lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl'); code=int(c['runtime_params']['noise']); ref=refs[c['base_task']]; tau=ref[2]; rq=torch.sqrt(tau.clamp_min(0)); rr_t=rr[c['base_task']]; cache={}; rows=[]
        for arm in ARMS:
            random.seed(es);np.random.seed(es);torch.manual_seed(es);ctypes.CDLL(None).srand(ctypes.c_uint(es)); env=OffScreenRenderEnv(bddl_file_name=str(LIB/c['bddl']),camera_heights=256,camera_widths=256,render_gpu_device_id=0); hook=None
            try:
                env.noise=0;env.seed(es);env.reset();obs=env.set_init_state(init)
                for _ in range(SETTLING_STEPS): obs=env.step(DUMMY_ACTION)[0]
                if arm!='frozen': hook=ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda mod,args,x,arm=arm: w2_apply(x,ref,arm,tau,rq,rr_t))
                gen=torch.Generator(device='cuda:0').manual_seed(ps); steps=0; success=False
                with torch.no_grad():
                    while steps<a.max_steps:
                        raw=observed(obs,code,ns,steps,dual,wrapper); key=(steps,array_hash(env.sim.get_state().flatten()),*(array_hash(np.asarray(raw[k])) for k in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'))); raw=match_pixels(raw,key,cache,[]); po=policy_observation(raw,lang,policy); z=torch.randn((1,12,7),device='cuda:0',generator=gen); _,rep=visual_from_obs(policy,po); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu().numpy()
                        for ac in plan:
                            obs,rew,done,info=env.step(ac); steps+=1; success=bool(is_success(env,info,rew,done))
                            if done or success or steps>=a.max_steps: break
                        if done or success: break
                rows.append({'case_id':c['case_id'],'base_task':c['base_task'],'group':g,'seed':s,'arm':arm,'success':success,'steps':steps}); print(json.dumps(rows[-1]),flush=True)
            finally:
                if hook: hook.remove()
                env.close()
        tmp=dest.with_suffix('.pending');tmp.write_text(json.dumps(rows));tmp.rename(dest)
if __name__=='__main__': main()
