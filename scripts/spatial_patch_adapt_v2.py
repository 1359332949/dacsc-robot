"""T3R-style zero-training agentview spatial patch suppression on V2 dev."""
from __future__ import annotations
import argparse,json,sys,os
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),'/hy-tmp/LIBERO-plus',str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
os.environ.setdefault('MUJOCO_GL','osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation  # noqa
from corp import load_policy  # noqa
from corp.representation.extractor import Representation  # noqa
from dacsc.differentiable_policy import sample_from_z0  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa
from dacsc.spatial_patch_bias import boundary_bg_weights,resampler_forward_with_bias  # noqa
from dacsc.static_visual_v2 import sensor_transform,array_hash  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa
import libero.libero.envs.bddl_utils as BDDLUtils  # noqa
import libero.libero.envs.env_wrapper as noise_wrapper  # noqa

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); ap.add_argument('--factors',default='clean'); ap.add_argument('--beta',type=float,default=2.0); ap.add_argument('--max-steps',type=int,default=220); ap.add_argument('--shard-id',type=int,default=0); ap.add_argument('--num-shards',type=int,default=1); ap.add_argument('--render-gpu',type=int,default=0); args=ap.parse_args()
 man=json.load(open(args.manifest)); res={}
 for p in Path(args.results).glob('*.json'):
  try: d=json.loads(p.read_text(errors='ignore')); res[d['case_id']]=d
  except Exception: pass
 factors=set(args.factors.split(',')); cases=[c for c in man['cases'] if c['factor'] in factors][args.shard_id::args.num_shards]
 policy=load_policy(args.checkpoint,device='cuda:0'); policy.eval(); from libero.libero.benchmark import grab_language_from_filename; rows=[]
 for c in cases:
  cr=res[c['case_id']]; initial=torch.load('/hy-tmp/LIBERO-plus/'+c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]; lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl'); env=OffScreenRenderEnv(bddl_file_name='/hy-tmp/LIBERO-plus/'+c['bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
  try:
   env.seed(cr['seeds']['environment_seed']); env.reset(); obs=env.set_init_state(initial)
   for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
   gen=torch.Generator(device='cuda:0'); gen.manual_seed(cr['seeds']['policy_flow_seed']); steps=0; success=False
   while steps<args.max_steps:
    ob=sensor_transform(obs,int(c['runtime_params']['noise']),cr['seeds']['sensor_noise_seed'],steps,noise_wrapper) if c['factor']=='sensor_noise' else obs
    if steps==0:
     for k in ('agentview','robot0_eye_in_hand'): exp=cr['initial_observation']['eye_in_hand' if 'eye' in k else 'agentview']['sha256']; act=array_hash(np.asarray(ob[k+'_image'])); assert act==exp,(c['case_id'],k,act,exp)
    po=policy_observation(ob,lang,policy); z0=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=gen); enc=encode_observation(policy,po,z0); bg=boundary_bg_weights(enc['patches']['agentview']); bias={'agentview':-args.beta*bg,'eye_in_hand':torch.zeros_like(bg)}
    visual=resampler_forward_with_bias(policy.representation_extractor.cross_attention_resampler,enc['patches'],enc['pooled_language'],bias); rep=Representation(tokens=torch.cat([visual,enc['language_tokens'].to(visual.dtype),enc['state_token']],dim=1),pooled_language=enc['pooled_language'],action_dtype=visual.dtype)
    with torch.no_grad(): ctrl=policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))
    done=False
    for k in range(12):
     obs,rew,done,info=env.step(np.clip(ctrl[0,k].detach().cpu().numpy(),-1,1)); steps+=1; success=bool(is_success(env,info,rew,done))
     if success or done or steps>=args.max_steps: break
    if success or done or steps>=args.max_steps: break
   rows.append({'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'success':success,'steps':steps,'method':'spatial_patch_bias','beta':args.beta}); print(json.dumps(rows[-1]),flush=True)
  finally: env.close()
 Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps({'schema':'spatial_patch_bias_v2','rows':rows},indent=2)+'\n')
if __name__=='__main__': main()
