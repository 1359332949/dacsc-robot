"""Local patch-energy TTA adapted to V2 100-case dev protocol (zero training)."""
from __future__ import annotations
import argparse,json,sys,os,hashlib
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),'/hy-tmp/LIBERO-plus',str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
os.environ.setdefault('MUJOCO_GL','osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation  # noqa
from corp import load_policy  # noqa
from corp.representation.extractor import Representation  # noqa
from target_only_action_validation import sample_z  # noqa
from dacsc.differentiable_policy import sample_from_z0  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa
from dacsc.static_visual_v2 import sensor_transform, array_hash  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa
import libero.libero.envs.bddl_utils as BDDLUtils  # noqa
import libero.libero.envs.env_wrapper as noise_wrapper  # noqa

def adapt(patches,means,eps=0.2):
 out={}
 for v,p in patches.items():
  a=p.float(); n=a.norm(dim=-1,keepdim=True).clamp_min(1e-6); target=means[v]
  factor=1.0+eps*torch.tanh(torch.log(torch.tensor(target,device=a.device)/(n+1e-6)))
  out[v]=(a*factor).to(p.dtype)
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); ap.add_argument('--factors',default='clean'); ap.add_argument('--eps',type=float,default=0.2); ap.add_argument('--means-json',default=''); ap.add_argument('--shard-id',type=int,default=0); ap.add_argument('--num-shards',type=int,default=1); ap.add_argument('--max-steps',type=int,default=220); ap.add_argument('--render-gpu',type=int,default=0); args=ap.parse_args()
 man=json.load(open(args.manifest)); res={}
 for p in Path(args.results).glob('*.json'):
  try:
   d=json.loads(p.read_text(errors='ignore')); res[d['case_id']]=d
  except Exception: pass
 factors=set(args.factors.split(',')); cases=[c for c in man['cases'] if c['factor'] in factors][args.shard_id::args.num_shards]
 policy=load_policy(args.checkpoint,device='cuda:0'); policy.eval(); from libero.libero.benchmark import grab_language_from_filename
 # calibrate patch norms on Clean 10 at t=0
 if args.means_json:
  means=json.load(open(args.means_json))
 else:
   means={}; cal=[c for c in man['cases'] if c['factor']=='clean']
   for c in cal:
    cr=res[c['case_id']]; initial=torch.load('/hy-tmp/LIBERO-plus/'+c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]; env=OffScreenRenderEnv(bddl_file_name='/hy-tmp/LIBERO-plus/'+c['bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
    try:
     env.seed(cr['seeds']['environment_seed']); env.reset(); obs=env.set_init_state(initial)
     for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
     lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl'); po=policy_observation(obs,lang,policy); z=torch.zeros((1,12,7),device='cuda:0'); enc=encode_observation(policy,po,z)
     for v,p in enc['patches'].items(): means.setdefault(v,[]).append(float(p.float().norm(dim=-1).mean()))
    finally: env.close()
   means={v:float(np.mean(x)) for v,x in means.items()}; print('means',means,flush=True)
 rows=[]
 for c in cases:
  cr=res[c['case_id']]; initial=torch.load('/hy-tmp/LIBERO-plus/'+c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]; lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl'); env=OffScreenRenderEnv(bddl_file_name='/hy-tmp/LIBERO-plus/'+c['bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
  try:
   env.seed(cr['seeds']['environment_seed']); env.reset(); obs=env.set_init_state(initial)
   for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
   gen=torch.Generator(device='cuda:0'); gen.manual_seed(cr['seeds']['policy_flow_seed']); steps=0; success=False
   while steps<args.max_steps:
    ob=sensor_transform(obs,int(c['runtime_params']['noise']),cr['seeds']['sensor_noise_seed'],steps,noise_wrapper) if c['factor']=='sensor_noise' else obs
    if steps==0:
     exp=cr['initial_observation']['agentview']['sha256']; act=array_hash(np.asarray(ob['agentview_image'])); assert act==exp,(c['case_id'],act,exp)
    po=policy_observation(ob,lang,policy); z0=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=gen); enc=encode_observation(policy,po,z0); patches=adapt(enc['patches'],means,args.eps)
    visual=policy.representation_extractor.cross_attention_resampler(patches,enc['pooled_language']); rep=Representation(tokens=torch.cat([visual,enc['language_tokens'].to(visual.dtype),enc['state_token']],dim=1),pooled_language=enc['pooled_language'],action_dtype=visual.dtype)
    with torch.no_grad(): ctrl=policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))
    done=False
    for k in range(12):
     obs,rew,done,info=env.step(np.clip(ctrl[0,k].detach().cpu().numpy(),-1,1)); steps+=1; success=bool(is_success(env,info,rew,done))
     if success or done or steps>=args.max_steps: break
    if success or done or steps>=args.max_steps: break
   rows.append({'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'success':success,'steps':steps,'method':'local_patch_energy','eps':args.eps}); print(json.dumps(rows[-1]),flush=True)
  finally: env.close()
 Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps({'schema':'local_patch_energy_v2','means':means,'rows':rows},indent=2)+'\n')
if __name__=='__main__': main()
