"""Build libero_10 init0 Block-3 calibration (10 Clean trajectories)."""
from __future__ import annotations
import argparse,json,sys,os,ctypes,random
from pathlib import Path
import numpy as np,torch
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'
from corp import load_policy
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation
from target_only_action_validation import sample_z,visual_from_obs
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import grab_language_from_filename
from dacsc.static_visual_v2 import seed_for
TASKS=['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it','KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it','KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it','KITCHEN_SCENE8_put_both_moka_pots_on_the_stove','LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket','LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket','LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket','LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate','LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate','STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy']
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task-index',type=int,required=True); ap.add_argument('--max-steps',type=int,default=520); a=ap.parse_args(); t=TASKS[a.task_index]
    bddl=LIB/'libero/libero/bddl_files/libero_10'/(t+'.bddl'); init=torch.load(LIB/'libero/libero/init_files/libero_10'/(t+'.pruned_init'),weights_only=False,map_location='cpu')[0]; lang=grab_language_from_filename('libero_10',t+'.bddl')
    policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False); ext=policy.representation_extractor; cap={}; h=ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x: cap.__setitem__('x',x[:,1:].detach()))
    es=seed_for('libero10_calib',t,'env'); ps=seed_for('libero10_calib',t,'policy'); random.seed(es);np.random.seed(es);torch.manual_seed(es);ctypes.CDLL(None).srand(ctypes.c_uint(es)); env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=0); ms=[]; qs=[]
    try:
        env.noise=0;env.seed(es);env.reset();obs=env.set_init_state(init)
        for _ in range(SETTLING_STEPS): obs=env.step(DUMMY_ACTION)[0]
        gen=torch.Generator(device='cuda:0').manual_seed(ps); steps=0
        with torch.no_grad():
            while steps<a.max_steps:
                po=policy_observation(obs,lang,policy); _,rep=visual_from_obs(policy,po); p=cap['x']; ms.append(p.mean(1).cpu()); qs.append(p.square().mean(1).cpu()); z=torch.randn((1,12,7),device='cuda:0',generator=gen); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu().numpy()
                for ac in plan:
                    obs,rew,done,info=env.step(ac); steps+=1
                    if done or steps>=a.max_steps: break
                if done or steps>=a.max_steps: break
    finally:
        h.remove(); env.close()
    OUT=ROOT/'runs/phase2_calib/libero10'; OUT.mkdir(parents=True,exist_ok=True); np.savez_compressed(OUT/f'task_{a.task_index}.npz',task=t,b3_m=torch.stack(ms).numpy(),b3_q=torch.stack(qs).numpy()); print(json.dumps({'task':t,'frames':len(ms),'steps':steps}),flush=True)
if __name__=='__main__': main()
