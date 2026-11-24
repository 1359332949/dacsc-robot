"""Frozen-only chunk-boundary physical/control anomaly recorder."""
from __future__ import annotations
import argparse,json,sys,os
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),'/hy-tmp/LIBERO-plus',str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
os.environ.setdefault('MUJOCO_GL','osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation  # noqa
from corp import load_policy  # noqa
from target_only_action_validation import sample_z,visual_from_obs  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa
from dacsc.static_visual_v2 import sensor_transform  # noqa
import libero.libero.envs.env_wrapper as noise_wrapper  # noqa

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); ap.add_argument('--factors',default='sensor_noise,background'); ap.add_argument('--max-steps',type=int,default=220); ap.add_argument('--shard-id',type=int,default=0); ap.add_argument('--num-shards',type=int,default=1); ap.add_argument('--render-gpu',type=int,default=0); args=ap.parse_args()
    man=json.load(open(args.manifest)); res={}
    for p in Path(args.results).glob('*.json'):
        try: d=json.loads(p.read_text(errors='ignore')); res[d['case_id']]=d
        except Exception: pass
    factors=set(args.factors.split(',')); cases=[c for c in man['cases'] if c['factor'] in factors][args.shard_id::args.num_shards]
    policy=load_policy(args.checkpoint,device='cuda:0'); policy.eval(); from libero.libero.benchmark import grab_language_from_filename
    rows=[]
    for c in cases:
        cr=res[c['case_id']]; seeds=cr['seeds']; lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl'); initial=torch.load('/hy-tmp/LIBERO-plus/'+c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]
        env=OffScreenRenderEnv(bddl_file_name='/hy-tmp/LIBERO-plus/'+c['bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        try:
            env.seed(seeds['environment_seed']); env.reset(); obs=env.set_init_state(initial)
            for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
            gen=torch.Generator(device='cuda:0'); gen.manual_seed(seeds['policy_flow_seed']); steps=0; success=False; chunks=[]
            while steps<args.max_steps:
                if c['factor']=='sensor_noise': ob=sensor_transform(obs,int(c['runtime_params']['noise']),seeds['sensor_noise_seed'],steps,noise_wrapper)
                else: ob=obs
                po=policy_observation(ob,lang,policy); _,rep=visual_from_obs(policy,po); s0=np.asarray(po['state'][0]).copy()
                z=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=gen); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].float().cpu().numpy(); plan=np.clip(plan,-1,1); begin=steps; done=False
                for k in range(12):
                    obs,rew,done,info=env.step(plan[k]); steps+=1; success=bool(is_success(env,info,rew,done))
                    if success or done or steps>=args.max_steps: break
                po2=policy_observation(obs,lang,policy); s1=np.asarray(po2['state'][0]).copy(); chunks.append({'start':begin,'end':steps,'state8_start':s0.tolist(),'state8_end':s1.tolist(),'action_norm':float(np.linalg.norm(plan[:max(1,steps-begin)])),'cmd_xyz_sum':plan[:max(1,steps-begin),:3].sum(0).tolist()})
                if success or done or steps>=args.max_steps: break
            rows.append({'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'success':success,'steps':steps,'chunks':chunks}); print(json.dumps({'case':c['case_id'],'factor':c['factor'],'success':success,'steps':steps}),flush=True)
        finally: env.close()
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps({'schema':'chunk_anomaly_v1','rows':rows},indent=2)+'\n')
if __name__=='__main__': main()
