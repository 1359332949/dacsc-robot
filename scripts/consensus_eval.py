"""Closed-loop zero-training consensus evaluation on frozen dev cases."""
from __future__ import annotations
import argparse,json,sys,os,time
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),'/hy-tmp/LIBERO-plus',str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
os.environ.setdefault('MUJOCO_GL','osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation  # noqa
from corp import load_policy  # noqa
from target_only_action_validation import sample_z, visual_from_obs  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa
import libero.libero.envs.bddl_utils as BDDLUtils  # noqa
from dacsc.static_visual_v2 import sensor_transform, seed_for  # noqa
import libero.libero.envs.env_wrapper as noise_wrapper  # noqa

def pair_dist(A,B):
    # A,B [12,7] normalized controller actions; physical-ish weights
    w=np.array([.05,.05,.05,.5,.5,.5,1.0]); d=(A-B)*w; return float(np.sqrt(np.mean(d*d)))
def consensus(cands,kind):
    C=np.stack(cands); M=len(C)
    if kind=='medoid':
        D=np.zeros((M,M))
        for i in range(M):
            for j in range(i+1,M): D[i,j]=D[j,i]=pair_dist(C[i],C[j])
        return C[int(np.argmin(D.sum(1)))]
    if kind=='trimmed':
        # drop farthest 25% from coordinate-wise median, average continuous, majority gripper
        med=np.median(C,axis=0); ds=np.array([np.mean((c-med)**2) for c in C]); keep=np.argsort(ds)[:max(2,int(round(M*.75)))]; K=C[keep]
        out=K[:,:,:6].mean(0); grip=np.sign(K[:,:,6].mean(0)); return np.concatenate([out,grip[...,None]],axis=-1)
    if kind=='z0zero': return C[0]
    raise ValueError(kind)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); ap.add_argument('--factors',default='clean'); ap.add_argument('--methods',default='frozen,medoid,trimmed,z0zero'); ap.add_argument('--m',type=int,default=8); ap.add_argument('--max-steps',type=int,default=220); ap.add_argument('--render-gpu',type=int,default=0); ap.add_argument('--shard-id',type=int,default=0); ap.add_argument('--num-shards',type=int,default=1); args=ap.parse_args()
    man=json.load(open(args.manifest)); res={}
    for p in Path(args.results).glob('*.json'):
        try: d=json.loads(p.read_text(errors='ignore')); res[d['case_id']]=d
        except Exception: pass
    factors=set(args.factors.split(',')); methods=args.methods.split(','); cases=[c for c in man['cases'] if c['factor'] in factors][args.shard_id::args.num_shards]
    policy=load_policy(args.checkpoint,device='cuda:0'); policy.eval(); out=[]
    from libero.libero.benchmark import grab_language_from_filename
    for c in cases:
        cr=res[c['case_id']]; seeds=cr['seeds']; initial=torch.load('/hy-tmp/LIBERO-plus/'+c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]; lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl')
        for method in methods:
            env=OffScreenRenderEnv(bddl_file_name='/hy-tmp/LIBERO-plus/'+c['bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            try:
                env.seed(seeds['environment_seed']); env.reset(); obs=env.set_init_state(initial)
                for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
                gen=torch.Generator(device='cuda:0'); gen.manual_seed(seeds['policy_flow_seed']); steps=0; success=False; chunks=[]
                while steps<args.max_steps:
                    obs_raw=obs
                    if c['factor']=='sensor_noise': obs_raw=sensor_transform(obs,int(c['runtime_params']['noise']),seeds['sensor_noise_seed'],steps,noise_wrapper)
                    po=policy_observation(obs_raw,lang,policy); _,rep=visual_from_obs(policy,po)
                    if method=='frozen':
                        z0=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=gen); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z0))[0].float().cpu().numpy(); plan=np.clip(plan,-1,1)
                    elif method=='z0zero':
                        z0=torch.zeros((1,12,7),device='cuda:0',dtype=torch.float32); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z0))[0].float().cpu().numpy(); plan=np.clip(plan,-1,1)
                    else:
                        cands=[]
                        for j in range(args.m):
                            g=torch.Generator(device='cuda:0'); g.manual_seed(seed_for(c['case_id'],method,steps,j)); z=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=g); a=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].float().cpu().numpy(); cands.append(np.clip(a,-1,1))
                        plan=consensus(cands,method)
                    s0=np.asarray(po['state'][0]); begin=steps; done=False
                    for k in range(12):
                        obs,rew,done,info=env.step(plan[k]); steps+=1; success=bool(is_success(env,info,rew,done));
                        if success or done or steps>=args.max_steps: break
                    s1=np.asarray(policy_observation(obs,lang,policy)['state'][0]) if False else None
                    chunks.append({'start':begin,'end':steps})
                    if success or done or steps>=args.max_steps: break
                out.append({'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'method':method,'success':success,'steps':steps,'chunks':chunks}); print(json.dumps(out[-1]),flush=True)
            except Exception as e:
                out.append({'case_id':c['case_id'],'factor':c['factor'],'method':method,'success':False,'error':type(e).__name__+': '+str(e)}); print(json.dumps(out[-1]),flush=True)
            finally: env.close()
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps({'schema':'consensus_eval_v1','m':args.m,'rows':out},indent=2)+'\n')
if __name__=='__main__': main()
