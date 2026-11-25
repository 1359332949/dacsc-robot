import argparse,json,sys,os,random,ctypes
from pathlib import Path
import numpy as np,torch
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'
from corp import load_policy
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation
from target_only_action_validation import visual_from_obs
from libero.libero.envs import OffScreenRenderEnv
from dacsc.static_visual_v2 import seed_for
from dacsc.single_block_probe import score
from dacsc.baselines_tta import Calib
from phase2_suite_eval import conditions
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=8);ap.add_argument('--shard',type=int,default=0);a=ap.parse_args();pol=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False);ext=pol.representation_extractor;cs=Calib(ROOT/'runs/phase2_calib/spatial_shared_calib.npz');ci=Calib(ROOT/'runs/phase2_calib/libero10/libero10_init0_calib.npz');cs.mu=cs.mu; cap={};h=ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:cap.__setitem__('x',x[:,1:].detach())); allc=[c for c in conditions() if c['group']=='clean']; a.out.mkdir(parents=True,exist_ok=True)
 for c in allc[a.shard::a.workers]:
  dest=a.out/f'{c["case_id"]}.json'
  if dest.exists():continue
  b=LIB/'libero/libero/bddl_files/libero_10'/c['bddl']; init=torch.load(LIB/'libero/libero/init_files/libero_10'/c['init'],weights_only=False,map_location='cpu')[c['init_index']]; lang=__import__('libero.libero.benchmark',fromlist=['grab_language_from_filename']).grab_language_from_filename('libero_10',c['base_task']+'.bddl'); es=seed_for('phase2_libero10',c['case_id'],'env'); random.seed(es);np.random.seed(es);torch.manual_seed(es);ctypes.CDLL(None).srand(ctypes.c_uint(es));env=OffScreenRenderEnv(bddl_file_name=str(b),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
  try:
   env.noise=0;env.seed(es);env.reset();obs=env.set_init_state(init)
   for _ in range(SETTLING_STEPS):obs=env.step(DUMMY_ACTION)[0]
   with torch.no_grad(): po=policy_observation(obs,lang,pol); visual_from_obs(pol,po); p=cap['x']; m=p.mean(1); sd=p.std(1,unbiased=False); rs={};ri={}
   for nm,cal in [('spatial',cs),('insuite',ci)]:
    mu,ss,tau=cal.ref(2); act=(score(m,sd,mu,ss)>tau).float().mean().item(); rs[nm]=act
   dest.write_text(json.dumps({'case_id':c['case_id'],'activation':rs}));print(json.dumps({'case_id':c['case_id'],**rs}),flush=True)
  finally: env.close()
 h.remove()
if __name__=='__main__':main()
