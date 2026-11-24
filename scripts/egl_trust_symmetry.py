"""EGL three-seed trust ablation and reversed-camera stress; actual rollouts for every arm."""
import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
from dacsc.static_visual_v2 import array_hash,file_hash,authorize,sensor_transform

VIEWS=('agentview','eye_in_hand'); METHODS=('rgb','trust','joint')

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
    from dacsc.symmetric_value_trust import symmetric_fit,solve,perturb
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['standard','wrist']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--seed',type=int,choices=(0,1,2),default=0);ap.add_argument('--limit',type=int);a=ap.parse_args()
    assert os.environ['MUJOCO_GL']==os.environ['PYOPENGL_PLATFORM']=='egl' and 0<=a.shard<a.workers
    final=ROOT/'docs/data/static_visual_v2_suite/final';mp=final/'dev_manifest.json';m=json.loads(mp.read_text());mh=file_hash(mp)
    assert mh=='d3e0d8c642cd215c879d2f0174663ff5ec585cd5c6d6b531a213c26e9fb982c0'; cases=m['cases']
    for c in cases:authorize(m,c,'low DOF Dev rollout',mh,mp)
    checkpoint=CORP/'checkpoints/corp_libero.pt'; assert file_hash(checkpoint)=='f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6';policy=load_policy(checkpoint,device='cuda:0').eval().requires_grad_(False)
    baseline=json.loads((ROOT/f'runs/egl_baseline_seed{a.seed}.json').read_text()); refs={r['case_id']:r for r in baseline['rows']};assert len(refs)==100
    a.out.mkdir(exist_ok=True,parents=True); assert a.mode!='wrist' or a.seed==0
    selected=sorted([c for c in cases if a.mode!='wrist' or c['factor']=='sensor_noise'],key=lambda c:c['case_id'])
    if a.limit:selected=selected[:a.limit]
    modes=('frozen','continuous','fixed','symmetric') if a.mode=='standard' else ('frozen','symmetric')
    for c in selected[a.shard::a.workers]:
        cid=c['case_id']; seeds=json.loads((final/'results'/(cid+'.json')).read_text())['seeds'];lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl'); initial=torch.load(LIB/c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]
        shared=None; pixel_cache={}  # reset per case; never match two different simulator states
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
                    raw=perturb(obs,int(c['runtime_params']['noise']),seeds['sensor_noise_seed'],steps,wrapper,a.mode=='wrist')
                    unchanged='agentview_image' if a.mode=='wrist' else 'robot0_eye_in_hand_image'
                    assert np.array_equal(raw[unchanged],obs[unchanged])
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
                    if steps==0 and method!='frozen':
                        base=encode_native(policy,po)
                        if method=='continuous':
                            shared=fit_gate(policy,base,seeds['method_augmentation_seed']);gate=shared
                        elif method=='fixed':gate=shared
                        else:gate=symmetric_fit(policy,base,seeds['method_augmentation_seed'])
                        assert gate is not None
                    logs=([0.,0.] if not gate or not gate['accepted'] else gate['logs'] if method=='symmetric' else [math.log(.10) if method=='fixed' else gate['candidate_log_trust'],0.])
                    z=torch.randn((1,12,7),device='cuda:0',generator=gen)
                    with torch.no_grad():
                        if min(logs)<0:
                            base=encode_native(policy,po);s2=torch.tensor(logs,device='cuda:0');rep=with_trust(base,s2[0]);
                            if method=='symmetric':x,_=solve(policy.action_generator,policy.action_generator.prepare_prefix(rep),z,s2,base.agentview_token_count)
                            else:x,_=integrate(policy,rep,z)
                        else:
                            _,rep=visual_from_obs(policy,po);x=sample_z(policy,rep,z)
                        plan=policy.normalizer.unnormalize_action(x)[0].clamp(-1,1).cpu().numpy()
                    trace.append(dict(step=steps,state=key[1],plan=array_hash(plan),images=[array_hash(np.asarray(po['images'][v])) for v in VIEWS]))
                    for action in plan:
                        ha.update(action.tobytes());obs,rew,done,info=env.step(action);steps+=1;success=bool(is_success(env,info,rew,done))
                        if done or success or steps>=220:break
                    if done or success:break

                row=dict(case_id=cid,seed=a.seed,method=method,base_task=c['base_task'],factor=c['factor'],sub_dimension=c.get('sub_dimension'),success=success,steps=steps,initial_sha=first,wrist_sha=wrist,action_sha=ha.hexdigest(),trace=trace,original_egl_first=unmatched_first,roundings=roundings,applied_logs=logs,regime=a.mode,gate=gate,seconds=time.monotonic()-started)
                row['archive_first_match']=first==refs[cid]['init_agentview_sha256']
                out.with_suffix('.pending').write_text(json.dumps(row,allow_nan=False))
                if method!='frozen':
                    ref_run=json.loads((a.out/f'{a.mode}_s{a.seed}_{cid}_frozen.json').read_text())
                    assert (first,wrist)==(ref_run['initial_sha'],ref_run['wrist_sha']),(cid,'paired initial pixels differ')
                    if not gate['accepted']:
                        against='frozen'
                        match_path=a.out/f'{a.mode}_s{a.seed}_{cid}_{against}.json'
                        if match_path.exists():
                            b=json.loads(match_path.read_text());assert (row['action_sha'],steps,success)==(b['action_sha'],b['steps'],b['success']),(cid,'rejected gate mismatch')
                out.with_suffix('.pending').rename(out); print(cid,method,success,steps,flush=True)
            finally:env.close()
if __name__=='__main__':main()
