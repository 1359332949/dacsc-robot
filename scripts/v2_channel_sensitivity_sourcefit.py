"""Source-only channel sensitivity on P0 50 source_fit Clean frames (10 tasks x5)."""
import argparse,json,sys,os,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np, torch, cv2
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),'/hy-tmp/LIBERO-plus',str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
os.environ.setdefault('MUJOCO_GL','osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation  # noqa
from corp import load_policy  # noqa
from corp.preprocessing import prepare_observation  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa
import libero.libero.envs.bddl_utils as BDDLUtils  # noqa
from libero.libero.benchmark import grab_language_from_filename  # noqa

def perturb(im,kind,rng):
 x=im.astype(np.float32)
 if kind=='bright_lo': x=x*0.8
 elif kind=='bright_hi': x=x*1.2
 elif kind=='contrast_lo': x=127.5+(x-127.5)*0.8
 elif kind=='contrast_hi': x=127.5+(x-127.5)*1.2
 elif kind=='blur': return cv2.GaussianBlur(im,(3,3),0)
 elif kind=='motion':
  k=np.zeros((3,3),np.float32); k[1,:]=1/3; return cv2.filter2D(im,-1,k)
 elif kind=='texture':
  noise=rng.normal(0,25,im.shape).astype(np.float32); return np.clip(x*0.7+noise*0.3,0,255).astype(np.uint8)
 return np.clip(x,0,255).astype(np.uint8)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--out',required=True); ap.add_argument('--max-tasks',type=int,default=10); args=ap.parse_args()
 man=json.load(open(args.manifest)); cases=[c for c in man['cases'] if c.get('role')=='source_fit' and c.get('factor')=='clean']
 by={}
 for c in cases: by.setdefault(c['base_task'],[]).append(c)
 tasks=[t for t in by if len(by[t])>=5][:args.max_tasks]
 cases=[c for t in tasks for c in by[t][:5]]
 print('source cases',len(cases),'tasks',len(tasks),flush=True)
 policy=load_policy(args.checkpoint,device='cuda:0'); policy.eval(); ex=policy.representation_extractor
 rng=np.random.RandomState(20261124); imgs=[]; states=[]; meta=[]
 for i,c in enumerate(cases):
  init=torch.load('/hy-tmp/LIBERO-plus/'+c['init_state_file'],map_location='cpu',weights_only=False)[c['init_state_index']]
  env=OffScreenRenderEnv(bddl_file_name='/hy-tmp/LIBERO-plus/'+c['clean_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=0)
  try:
   env.seed(7777); env.reset(); obs=env.set_init_state(init)
   for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
   lang=grab_language_from_filename(c['suite'],c['base_task']+'.bddl'); po=policy_observation(obs,lang,policy); z=torch.zeros((1,12,7),device='cuda:0'); enc=encode_observation(policy,po,z)
   imgs.append(cv2.resize(np.asarray(obs['agentview_image']).astype(np.uint8),(224,224),interpolation=cv2.INTER_AREA)); states.append(np.asarray(po['state'][0] if False else obs['robot0_eef_pos'],dtype=np.float32)); meta.append({'case_id':c['case_id'],'task':c['base_task'],'index':c['init_state_index'],'role':c['role']})
   print(i,c['base_task'][:20],c['init_state_index'],flush=True)
  finally: env.close()
 # build perturbed patches from clean image? We need images; re-render is expensive. Instead perturb patch tokens? Domain sensitivity on patches under image perturbations requires images. Use synthetic patch-space noise is not valid.
 # encode clean and perturbed images in batches
 kinds=['clean','bright_lo','bright_hi','contrast_lo','contrast_hi','blur','motion','texture']; P={}
 imgs=np.stack(imgs); state=np.zeros((len(imgs),8),np.float32)
 for kind in kinds:
  arr=np.stack([perturb(im,kind,rng) if kind!='clean' else im for im in imgs]); obs={'images':{'agentview':arr,'eye_in_hand':np.zeros_like(arr)},'state':state,'instruction':['x']*len(arr)}; prep=prepare_observation(obs,config=policy.config,normalizer=policy.normalizer,device=torch.device('cuda:0'))
  with torch.no_grad(): pp=ex.visual_encoder(prep['images']['agentview'])
  P[kind]=pp.float().cpu().numpy(); print('encoded',kind,flush=True)
 base=P['clean']; dom=np.stack([P[k] for k in kinds if k!='clean']); s_dom=np.abs(dom-base[None]).mean(axis=0).mean(axis=(0,1)); s_task=base.std(axis=0).mean(axis=0); R=s_dom/(s_task+1e-6); mu=base.mean(axis=(0,1)); order=np.argsort(-R); top=order[:max(1,int(.2*384))].tolist()
 out={'schema':'v2_channel_sensitivity_source_v2','pairs':args.manifest,'n_images':len(imgs),'tasks':tasks,'cases':meta,'perturbations':kinds,'s_domain':s_dom.tolist(),'s_task':s_task.tolist(),'R':R.tolist(),'channel_mean':mu.tolist(),'top20pct':top,'target_dev_images_read':False}
 Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(out,indent=2)+'\n'); print('top',top[:10])
if __name__=='__main__': main()
