"""Frozen seed-0 Dev experiment. Calibrate on EGL, run independent resets per arm."""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
from dacsc.static_visual_v2 import array_hash,file_hash,authorize,sensor_transform

VIEWS=('agentview','eye_in_hand'); METHODS=('rgb','trust','joint')

def moments(image):
    x=np.asarray(image,dtype='float64').reshape(-1,3)
    return np.stack([x.mean(0),np.square(x).mean(0)])

def rgb_fold(stats,case):
    refs=[r for r in stats if r['base_task']!=case['base_task']]
    if len(refs)!=9 or len({r['base_task'] for r in refs})!=9 or any(r['factor']!='clean' for r in refs): raise ValueError('invalid Clean fold')
    return {v:np.mean([r[v] for r in refs],axis=0) for v in VIEWS}

def correct(image,m):
    x=np.asarray(image,dtype=np.float64);mu=x.mean((0,1));sd=x.std((0,1));ref,second=m
    target=(x-mu)*(np.maximum(second-ref**2,0)**.5 / np.maximum(sd,1e-6))+ref
    return np.clip(np.rint(.5*x+.5*target),0,255).astype('uint8')

def main():
    import torch
    from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation
    from corp import load_policy
    from target_only_action_validation import sample_z,visual_from_obs
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.benchmark import grab_language_from_filename
    import libero.libero.envs.env_wrapper as wrapper
    from dacsc.flow_value_trust import encode_native,fit_gate,with_trust
    from dacsc.flow_repair_kv import integrate
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['stats','pilot','run']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--seed',type=int,choices=(0,1,2),default=0);ap.add_argument('--method',choices=METHODS);a=ap.parse_args()
    assert os.environ['MUJOCO_GL']==os.environ['PYOPENGL_PLATFORM']=='egl' and 0<=a.shard<a.workers
    final=ROOT/'docs/data/static_visual_v2_suite/final';mp=final/'dev_manifest.json';m=json.loads(mp.read_text());mh=file_hash(mp)
    assert mh=='d3e0d8c642cd215c879d2f0174663ff5ec585cd5c6d6b531a213c26e9fb982c0'; cases=m['cases']
    for c in cases:authorize(m,c,'low DOF Dev rollout',mh,mp)
    checkpoint=CORP/'checkpoints/corp_libero.pt'; assert file_hash(checkpoint)=='f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6';policy=load_policy(checkpoint,device='cuda:0').eval().requires_grad_(False)
    baseline=json.loads((ROOT/f'runs/egl_baseline_seed{a.seed}.json').read_text()); refs={r['case_id']:r for r in baseline['rows']};assert len(refs)==100
    a.out.mkdir(exist_ok=True,parents=True);sp=a.out/'clean_moments.json';stats=[] if a.mode=='stats' else json.loads(sp.read_text())
    selected=sorted(cases,key=lambda c:c['case_id']);modes=('frozen',)+METHODS if not a.method else ('frozen',a.method)
    if a.mode=='stats': selected=[c for c in selected if c['factor']=='clean'];modes=('stats',)
    if a.mode=='pilot':selected=[next(c for c in selected if c['factor']==f) for f in ('clean','sensor_noise','background','lighting')];modes=('frozen','rgb','trust','joint')
    for c in selected[a.shard::a.workers]:
        cid=c['case_id']; seeds=json.loads((final/'results'/(cid+'.json')).read_text())['seeds'];lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl'); initial=torch.load(LIB/c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]
        pixel_cache={}  # reset per case; never match two different simulator states
        for method in modes:
            out=a.out/f'{a.mode}_s{a.seed}_{cid}_{method}.json';assert not out.exists();started=time.monotonic();gate=None;steps=0;ha=hashlib.sha256();first=None;trace=[];roundings=[]
            import random,ctypes
            env_seed=int(seeds['environment_seed']);random.seed(env_seed);np.random.seed(env_seed)
            torch.manual_seed(env_seed);ctypes.CDLL(None).srand(ctypes.c_uint(env_seed))
            env=OffScreenRenderEnv(bddl_file_name=str(LIB/c['bddl']),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
            try:
                env.noise=0;env.seed(seeds['environment_seed']);env.reset();obs=env.set_init_state(initial)
                for _ in range(SETTLING_STEPS):obs,_,_,_=env.step(DUMMY_ACTION)
                from dacsc.static_visual_v2 import seed_for
                gen=torch.Generator(device='cuda:0').manual_seed(seeds['policy_flow_seed'] if not a.seed else seed_for(c['pair_group_id'],a.seed,'egl_policy_seed'))
                while steps<220:
                    raw=sensor_transform(obs,int(c['runtime_params']['noise']),seeds['sensor_noise_seed'],steps,wrapper)
                    keys=('agentview_image','robot0_eye_in_hand_image')
                    key=(steps,array_hash(env.sim.get_state().flatten()),array_hash(np.asarray(raw['robot0_eef_pos'])),array_hash(np.asarray(raw['robot0_eef_quat'])),array_hash(np.asarray(raw['robot0_gripper_qpos'])))
                    if steps==0:unmatched_first=array_hash(np.asarray(raw[keys[0]]))
                    if key in pixel_cache:
                        raw=dict(raw)
                        for k,img in zip(keys,pixel_cache[key]):
                            seen=np.asarray(raw[k]);diff=np.abs(seen.astype(float)-img.astype(float));n=int((diff.max(axis=-1)>0).sum());mx=float(diff.max())
                            if n:
                                assert mx<=1 and n<=64,(cid,steps,k,mx,n,'not rounding-level variation')
                                roundings.append([steps,k,n,mx]);raw[k]=img
                    else:pixel_cache[key]=tuple(np.asarray(raw[k]).copy() for k in keys)
                    if steps==0:
                        first=array_hash(np.asarray(raw['agentview_image']))
                        wrist=array_hash(np.asarray(raw['robot0_eye_in_hand_image']))
                    po=policy_observation(raw,lang,policy)
                    if method=='stats':
                        stats.append(dict(case_id=cid,base_task=c['base_task'],factor='clean',initial_sha=first,**{v:moments(po['images'][v][0]).tolist() for v in VIEWS}));break
                    if method in ('rgb','joint'):
                        fold=rgb_fold(stats,c);po=dict(po,images={v:torch.from_numpy(correct(po['images'][v][0],fold[v])).unsqueeze(0) for v in VIEWS})
                    if steps==0 and method in ('trust','joint'):
                        base=encode_native(policy,po);gate=fit_gate(policy,base,seeds['method_augmentation_seed'])
                    z=torch.randn((1,12,7),device='cuda:0',generator=gen)
                    with torch.no_grad():
                        if gate and gate['applied_log_trust']!=0:
                            base=encode_native(policy,po);x,_=integrate(policy,with_trust(base,torch.tensor(gate['applied_log_trust'],device='cuda:0')),z)
                        else:
                            _,rep=visual_from_obs(policy,po);x=sample_z(policy,rep,z)
                        plan=policy.normalizer.unnormalize_action(x)[0].clamp(-1,1).cpu().numpy()
                    trace.append(dict(step=steps,state=key[1],plan=array_hash(plan),images=[array_hash(np.asarray(po['images'][v])) for v in VIEWS]))
                    for action in plan:
                        ha.update(action.tobytes());obs,rew,done,info=env.step(action);steps+=1;success=bool(is_success(env,info,rew,done))
                        if done or success or steps>=220:break
                    if done or success:break
                if method=='stats':continue
                row=dict(case_id=cid,seed=a.seed,method=method,base_task=c['base_task'],factor=c['factor'],sub_dimension=c.get('sub_dimension'),success=success,steps=steps,initial_sha=first,wrist_sha=wrist,action_sha=ha.hexdigest(),trace=trace,original_egl_first=unmatched_first,roundings=roundings,gate=gate,seconds=time.monotonic()-started)
                row['archive_first_match']=first==refs[cid]['init_agentview_sha256']
                out.with_suffix('.pending').write_text(json.dumps(row,allow_nan=False))
                if method!='frozen':
                    ref_run=json.loads((a.out/f'{a.mode}_s{a.seed}_{cid}_frozen.json').read_text())
                    assert (first,wrist)==(ref_run['initial_sha'],ref_run['wrist_sha']),(cid,'paired initial pixels differ')
                    if method in ('trust','joint') and not gate['accepted']:
                        against='frozen' if method=='trust' else 'rgb'
                        match_path=a.out/f'{a.mode}_s{a.seed}_{cid}_{against}.json'
                        if match_path.exists():
                            b=json.loads(match_path.read_text());assert (row['action_sha'],steps,success)==(b['action_sha'],b['steps'],b['success']),(cid,'rejected gate mismatch')
                out.with_suffix('.pending').rename(out); print(cid,method,success,steps,flush=True)
            finally:env.close()
    if a.mode=='stats':assert len(stats)==10;sp.write_text(json.dumps(stats,indent=2)+'\n')
if __name__=='__main__':main()
