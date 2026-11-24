"""Frozen CoRP baseline under EGL/GPU rendering, exact dev manifest cases."""
from __future__ import annotations
import argparse,json,sys,os
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),'/hy-tmp/LIBERO-plus',str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation  # noqa
from corp import load_policy  # noqa
from target_only_action_validation import sample_z, visual_from_obs  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa
from libero.libero.benchmark import grab_language_from_filename  # noqa
from dacsc.static_visual_v2 import sensor_transform, seed_for, array_hash  # noqa
import libero.libero.envs.env_wrapper as noise_wrapper  # noqa

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); ap.add_argument('--seed-index',type=int,default=0); ap.add_argument('--max-steps',type=int,default=220); ap.add_argument('--shard-id',type=int,default=0); ap.add_argument('--num-shards',type=int,default=1); ap.add_argument('--render-gpu',type=int,default=0); a=ap.parse_args()
    man=json.load(open(a.manifest)); cases=man['cases'][a.shard_id::a.num_shards]
    res={}
    for p in Path(a.results).glob('*.json'):
        try:
            d=json.loads(p.read_text(errors='ignore')); res[d['case_id']]=d
        except Exception: pass
    policy=load_policy(a.checkpoint,device='cuda:0'); policy.eval()
    out=[]; Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    for c in cases:
        initial=torch.load('/hy-tmp/LIBERO-plus/'+c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]; cr=res[c['case_id']]; seeds=cr['seeds']
        env=OffScreenRenderEnv(bddl_file_name='/hy-tmp/LIBERO-plus/'+c['bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=a.render_gpu)
        try:
            native_noise=int(env.noise)
            env.noise=0  # manual step-keyed sensor_transform owns corruption
            if int(env.noise)!=0:raise RuntimeError('wrapper noise disable failed')
            env.seed(seeds['environment_seed'])
            env.reset(); obs=env.set_init_state(initial)
            noise=int(c['runtime_params']['noise'])
            for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
            raw0=obs
            init_sha=None
            lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl')
            gen=torch.Generator(device='cuda:0')
            if a.seed_index==0: gen.manual_seed(int(seeds['policy_flow_seed']))
            else: gen.manual_seed(seed_for(c['pair_group_id'],a.seed_index,'egl_policy_seed'))
            steps=0; success=False; chunks=[]
            while steps<a.max_steps:
                obs_raw=raw0
                if c['factor']=='sensor_noise': obs_q=sensor_transform(obs_raw,noise,seeds['sensor_noise_seed'],steps,noise_wrapper)
                else: obs_q=obs_raw
                if init_sha is None:init_sha=array_hash(np.asarray(obs_q['agentview_image']))
                po=policy_observation(obs_q,lang,policy); _,rep=visual_from_obs(policy,po)
                z0=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=gen)
                plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z0))[0].float().cpu().numpy(); plan=np.clip(plan,-1,1)
                begin=steps; done=False
                for k in range(12):
                    obs,rew,done,info=env.step(plan[k]); steps+=1; success=bool(is_success(env,info,rew,done)); raw0=obs
                    if success or done or steps>=a.max_steps: break
                chunks.append({'start':begin,'end':steps})
                if success or done or steps>=a.max_steps: break
            out.append({'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'success':success,'steps':steps,'init_agentview_sha256':init_sha,'seed_index':a.seed_index,'chunks':chunks,'native_noise_before_reset':native_noise})
            print(json.dumps(out[-1]),flush=True)
        except Exception as e:
            out.append({'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'success':False,'error':type(e).__name__+': '+str(e),'seed_index':a.seed_index}); print(json.dumps(out[-1]),flush=True)
        finally: env.close()
    Path(a.out).write_text(json.dumps({'schema':'frozen_egl_baseline_v1','render':'egl','seed_index':a.seed_index,'rows':out},indent=2)+'\n')
if __name__=='__main__': main()
