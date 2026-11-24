"""EGL closed-loop consensus for selected zero-training operators, 3 policy seeds."""
from __future__ import annotations
import argparse,json,sys,os
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),'/hy-tmp/LIBERO-plus',str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation
from corp import load_policy
from target_only_action_validation import sample_z, visual_from_obs
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import grab_language_from_filename
from dacsc.static_visual_v2 import sensor_transform, seed_for
import libero.libero.envs.env_wrapper as noise_wrapper

def pair_dist(A,B):
    w=np.array([.05,.05,.05,.5,.5,.5,1.0]); d=(A-B)*w; return float(np.sqrt(np.mean(d*d)))
def consensus(C,kind):
    C=np.stack(C); M=len(C)
    if kind=='trimmed':
        med=np.median(C,axis=0); ds=np.array([np.mean((c-med)**2) for c in C]); keep=np.argsort(ds)[:max(2,int(round(M*.75)))]; K=C[keep]
        out=K[:,:,:6].mean(0); grip=np.sign(K[:,:,6].mean(0)); return np.concatenate([out,grip[...,None]],axis=-1)
    if kind=='z0zero': return C[0]
    raise ValueError(kind)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); ap.add_argument('--methods',default='z0zero,trimmed'); ap.add_argument('--seed-index',type=int,default=0); ap.add_argument('--m',type=int,default=8); ap.add_argument('--max-steps',type=int,default=220); ap.add_argument('--shard-id',type=int,default=0); ap.add_argument('--num-shards',type=int,default=1); a=ap.parse_args()
    man=json.load(open(a.manifest)); cases=man['cases'][a.shard_id::a.num_shards]
    res={}
    for p in Path(a.results).glob('*.json'):
        try:
            d=json.loads(p.read_text(errors='ignore')); res[d['case_id']]=d
        except Exception: pass
    policy=load_policy(a.checkpoint,device='cuda:0'); policy.eval(); out=[]; methods=a.methods.split(',')
    for c in cases:
        cr=res[c['case_id']]; seeds=cr['seeds']; initial=torch.load('/hy-tmp/LIBERO-plus/'+c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]; lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl'); noise=int(c['runtime_params']['noise'])
        for method in methods:
            env=OffScreenRenderEnv(bddl_file_name='/hy-tmp/LIBERO-plus/'+c['bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=0)
            try:
                native_noise=int(env.noise)
                env.noise=0  # manual step-keyed sensor_transform owns corruption
                if int(env.noise)!=0:raise RuntimeError('wrapper noise disable failed')
                env.seed(seeds['environment_seed']); env.reset(); obs=env.set_init_state(initial)
                for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
                raw0=obs; steps=0; success=False; chunks=[]; init_sha=None
                # policy seed: seed0 uses exact baseline flow seed; later seeds use independent stream
                gen=torch.Generator(device='cuda:0')
                gen.manual_seed(int(seeds['policy_flow_seed']) if a.seed_index==0 else seed_for(c['pair_group_id'],a.seed_index,'egl_policy_seed'))
                while steps<a.max_steps:
                    obs_q=sensor_transform(raw0,noise,seeds['sensor_noise_seed'],steps,noise_wrapper) if c['factor']=='sensor_noise' else raw0
                    if init_sha is None: init_sha=__import__('hashlib').sha256(np.ascontiguousarray(np.asarray(obs_q['agentview_image'])).tobytes()).hexdigest()
                    po=policy_observation(obs_q,lang,policy); _,rep=visual_from_obs(policy,po)
                    if method=='z0zero':
                        z=torch.zeros((1,12,7),device='cuda:0',dtype=torch.float32); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].float().cpu().numpy(); plan=np.clip(plan,-1,1)
                    else:
                        cands=[]
                        for j in range(a.m):
                            g=torch.Generator(device='cuda:0'); g.manual_seed(seed_for(c['case_id'],method,steps,j,a.seed_index)); z=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=g); cands.append(np.clip(policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].float().cpu().numpy(),-1,1))
                        plan=consensus(cands,method)
                    begin=steps; done=False
                    for k in range(12):
                        obs,rew,done,info=env.step(plan[k]); steps+=1; success=bool(is_success(env,info,rew,done)); raw0=obs
                        if success or done or steps>=a.max_steps: break
                    chunks.append({'start':begin,'end':steps})
                    if success or done or steps>=a.max_steps: break
                out.append({'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'method':method,'seed_index':a.seed_index,'success':success,'steps':steps,'init_agentview_sha256':init_sha,'chunks':chunks,'native_noise_before_reset':native_noise})
                print(json.dumps(out[-1]),flush=True)
            except Exception as e:
                out.append({'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'method':method,'seed_index':a.seed_index,'success':False,'error':type(e).__name__+': '+str(e)}); print(json.dumps(out[-1]),flush=True)
            finally: env.close()
    Path(a.out).write_text(json.dumps({'schema':'egl_consensus_v1','render':'egl','seed_index':a.seed_index,'rows':out},indent=2)+'\n')
if __name__=='__main__': main()
