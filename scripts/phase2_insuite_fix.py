"""Frozen vs in-suite V1 on Clean/Lighting/Background with fixed language and true sub_group."""
from __future__ import annotations
import argparse,json,sys,os,ctypes,random
from pathlib import Path
import numpy as np,torch
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'
from corp import load_policy
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation
from target_only_action_validation import sample_z,visual_from_obs
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import grab_language_from_filename
from dacsc.single_block_probe import align
from dacsc.static_visual_v2 import array_hash,seed_for
from dacsc.baselines_tta import Calib
from egl_shallow_block3_closed_loop import observed,match_pixels
import libero.libero.envs.env_wrapper as wrapper
from phase2_suite_eval import conditions
ARMS=['frozen','ours_v1_insuite']
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--workers',type=int,default=16); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--max-steps',type=int,default=520); ap.add_argument('--groups',default='clean,lighting,background'); a=ap.parse_args(); policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False); ext=policy.representation_extractor; cal=Calib(ROOT/'runs/phase2_calib/libero10/libero10_init0_calib.npz'); allc=[c for c in conditions() if c['group'] in a.groups.split(',')]; print('conditions',len(allc),flush=True); (a.out/'trials').mkdir(parents=True,exist_ok=True)
    for c in allc[a.shard::a.workers]:
        dest=a.out/'trials'/f'{c["case_id"]}.json'
        if dest.exists(): continue
        bddl=LIB/'libero/libero/bddl_files/libero_10'/c['bddl']; init=torch.load(LIB/'libero/libero/init_files/libero_10'/c['init'],weights_only=False,map_location='cpu')[c['init_index']]; lang=grab_language_from_filename('libero_10',c['base_task']+'.bddl'); assert lang==grab_language_from_filename('libero_10',c['base_task']+'.bddl') and not any(ch.isdigit() for ch in lang); es=seed_for('phase2_libero10',c['case_id'],'env'); ps=seed_for('phase2_libero10',c['case_id'],'policy'); ns=seed_for('phase2_libero10',c['case_id'],'sensor'); cache={}; rows=[]
        for arm in ARMS:
            random.seed(es);np.random.seed(es);torch.manual_seed(es);ctypes.CDLL(None).srand(ctypes.c_uint(es)); env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=0); hook=None
            try:
                env.noise=0;env.seed(es);env.reset();obs=env.set_init_state(init)
                for _ in range(SETTLING_STEPS): obs=env.step(DUMMY_ACTION)[0]
                if arm=='ours_v1_insuite': hook=ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align(x,cal.ref(2),.75,'dead'))
                gen=torch.Generator(device='cuda:0').manual_seed(ps); steps=0; success=False; finite=True
                with torch.no_grad():
                    while steps<a.max_steps:
                        raw=observed(obs,c['code'],ns,steps,c['dual'],wrapper); key=(steps,array_hash(env.sim.get_state().flatten()),*(array_hash(np.asarray(raw[k])) for k in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'))); raw=match_pixels(raw,key,cache,[]); po=policy_observation(raw,lang,policy); z=torch.randn((1,12,7),device='cuda:0',generator=gen); _,rep=visual_from_obs(policy,po); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu().numpy(); finite=finite and bool(np.isfinite(plan).all())
                        for ac in plan:
                            obs,rew,done,info=env.step(ac); steps+=1; success=bool(is_success(env,info,rew,done))
                            if done or success or steps>=a.max_steps: break
                        if done or success: break
                rows.append({'case_id':c['case_id'],'group':c['group'],'sub_group':c.get('sub_group'),'base_task':c['base_task'],'init_index':c['init_index'],'arm':arm,'success':success,'steps':steps,'finite':finite})
            finally:
                if hook: hook.remove()
                env.close()
        tmp=dest.with_suffix('.pending');tmp.write_text(json.dumps(rows));tmp.rename(dest); print(json.dumps({'case_id':c['case_id'],'group':c['group'],'sub_group':c.get('sub_group'),'rows':rows}),flush=True)
if __name__=='__main__': main()
