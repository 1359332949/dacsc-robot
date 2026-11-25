"""Phase 1.5 Clean sanity on libero_spatial init_index=1 for 5 arms."""
from __future__ import annotations
import json,sys,os,ctypes,random,argparse
from pathlib import Path
import numpy as np,torch
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'
from corp import load_policy
from corp.preprocessing import prepare_observation
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation
from target_only_action_validation import sample_z,visual_from_obs
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import grab_language_from_filename
from dacsc.static_visual_v2 import seed_for
from dacsc.baselines_tta import Calib,pixel_align_raw,ActMAD
from dacsc.single_block_probe import align
ARMS=['frozen','v1_block3','pixel_align','deep_block12','actmad_true']
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task-index',type=int,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--max-steps',type=int,default=300); ap.add_argument('--lr',type=float,default=2.5e-4); a=ap.parse_args()
    man=json.load(open(ROOT/'docs/data/static_visual_v2_suite/final/dev_manifest.json')); cleans=[c for c in man['cases'] if c['factor']=='clean']; c=cleans[a.task_index]; task=c['base_task']; bddl=LIB/c['bddl']; init=torch.load(LIB/c['init_state_file'],weights_only=False,map_location='cpu')[1]; lang=grab_language_from_filename('libero_spatial',task+'.bddl')
    policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False); cal=Calib(ROOT/'runs/phase2_calib/spatial_shared_calib.npz'); rows=[]
    for arm in ARMS:
        es=seed_for('phase15_sanity',task,arm,'env'); ps=seed_for('phase15_sanity',task,arm,'policy'); random.seed(es);np.random.seed(es);torch.manual_seed(es);ctypes.CDLL(None).srand(ctypes.c_uint(es)); env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=0); hook=None; ada=None
        try:
            env.noise=0;env.seed(es);env.reset();obs=env.set_init_state(init)
            for _ in range(SETTLING_STEPS): obs=env.step(DUMMY_ACTION)[0]
            if arm=='v1_block3': hook=policy.representation_extractor.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align(x,cal.ref(2),.75,'dead'))
            elif arm=='deep_block12': hook=policy.representation_extractor.visual_encoder.backbone.blocks[11].register_forward_hook(lambda m,a,x:align(x,cal.ref(11),.75,'dead'))
            elif arm=='actmad_true': ada=ActMAD(policy,cal,lr=a.lr); ada.reset()
            gen=torch.Generator(device='cuda:0').manual_seed(ps); steps=0; success=False; finite=True
            with torch.no_grad():
                while steps<a.max_steps:
                    po=policy_observation(obs,lang,policy); raw=po
                    if arm=='pixel_align': raw=pixel_align_raw(po,cal)
                    if ada is not None:
                        batch=prepare_observation(po,config=policy.config,normalizer=policy.normalizer,device='cuda:0'); ada.update(batch)
                    z=torch.randn((1,12,7),device='cuda:0',generator=gen); _,rep=visual_from_obs(policy,raw); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu().numpy(); finite=finite and bool(np.isfinite(plan).all())
                    for ac in plan:
                        obs,rew,done,info=env.step(ac); steps+=1; success=bool(is_success(env,info,rew,done))
                        if done or success or steps>=a.max_steps: break
                    if done or success: break
            rows.append({'task':task,'arm':arm,'success':success,'steps':steps,'finite':finite,'init_index':1,'lr':a.lr if arm=='actmad_true' else None})
        finally:
            if hook: hook.remove()
            if ada: ada.close()
            env.close()
        print(json.dumps(rows[-1]),flush=True)
    a.out.mkdir(parents=True,exist_ok=True); (a.out/f'{a.task_index}.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__': main()
