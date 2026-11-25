"""Phase 2A: 5-arm fair physical evaluation on libero_10."""
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
from dacsc.baselines_tta import Calib,pixel_align_raw,ActMAD
from egl_shallow_block3_closed_loop import observed,match_pixels
import libero.libero.envs.env_wrapper as wrapper
ARMS=['frozen','pixel_align','deep_block12','actmad_true','ours_v1']
TASKS=['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it','KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it','KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it','KITCHEN_SCENE8_put_both_moka_pots_on_the_stove','LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket','LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket','LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket','LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate','LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate','STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy']
FAMS=['motion_blur','gaussian_blur','zoom_blur','fog','glass_blur']
def conditions():
    out=[]
    for ti,t in enumerate(TASKS):
        for k in range(1,11): out.append({'case_id':f'clean_{ti}_{k}','group':'clean','base_task':t,'bddl':t+'.bddl','init':t+'.pruned_init','init_index':k,'code':0,'dual':False})
        for fam_i,fam in enumerate(FAMS):
            for k in (1,2): out.append({'case_id':f'sensor_{ti}_{fam}_{k}','group':'sensor_bilateral','base_task':t,'bddl':t+'.bddl','init':t+'.pruned_init','init_index':k,'code':fam_i*10+5,'dual':True})
        lights=sorted([f.stem for f in (LIB/'libero/libero/bddl_files/libero_10').glob(t+'_light_*.bddl')])[:10]
        tabs=sorted([f.stem for f in (LIB/'libero/libero/bddl_files/libero_10').glob(t+'*table_*.bddl')]); scns=sorted([f.stem for f in (LIB/'libero/libero/bddl_files/libero_10').glob(t+'*scene_*.bddl')]); bgs=[]
        for j in range(5):
            if j<len(tabs): bgs.append((tabs[j],'table'))
            if j<len(scns): bgs.append((scns[j],'scene'))
        while len(bgs)<10:
            pool=[(x,'table') for x in tabs] + [(x,'scene') for x in scns]
            bgs.append(pool[len(bgs)%len(pool)])
        bgs=bgs[:10]
        for j,v in enumerate(lights): out.append({'case_id':f'light_{ti}_{j}','group':'lighting','base_task':t,'bddl':v+'.bddl','init':t+'.pruned_init','init_index':(j%10)+1,'code':0,'dual':False})
        for j,(v,sg) in enumerate(bgs): out.append({'case_id':f'bg_{ti}_{j}','group':'background','sub_group':sg,'base_task':t,'bddl':v+'.bddl','init':t+'.pruned_init','init_index':(j%10)+1,'code':0,'dual':False})
    assert len(out)==400 and all(sum(1 for x in out if x['group']==g)==100 for g in ('clean','sensor_bilateral','lighting','background')),(len(out),{g:sum(1 for x in out if x['group']==g) for g in ('clean','sensor_bilateral','lighting','background')})
    return out
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--workers',type=int,default=8); ap.add_argument('--shard',type=int,default=0); ap.add_argument('--max-steps',type=int,default=520); ap.add_argument('--case-ids',default=None); a=ap.parse_args()
    policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False); ext=policy.representation_extractor; cal=Calib(ROOT/'runs/phase2_calib/spatial_shared_calib.npz'); allc=conditions(); print('conditions',len(allc),flush=True)
    if a.case_ids: ids=set(a.case_ids.split(',')); allc=[c for c in allc if c['case_id'] in ids]; print('selected',len(allc),flush=True)
    (a.out/'trials').mkdir(parents=True,exist_ok=True)
    for c in allc[a.shard::a.workers]:
        dest=a.out/'trials'/f'{c["case_id"]}.json'
        if dest.exists(): continue
        bddl=LIB/'libero/libero/bddl_files/libero_10'/c['bddl']; initf=LIB/'libero/libero/init_files/libero_10'/c['init']; init=torch.load(initf,weights_only=False,map_location='cpu')[c['init_index']]; lang=grab_language_from_filename('libero_10',c['base_task']+'.bddl'); es=seed_for('phase2_libero10',c['case_id'],'env'); ps=seed_for('phase2_libero10',c['case_id'],'policy'); ns=seed_for('phase2_libero10',c['case_id'],'sensor'); cache={}; rows=[]
        for arm in ARMS:
            random.seed(es);np.random.seed(es);torch.manual_seed(es);ctypes.CDLL(None).srand(ctypes.c_uint(es)); env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=0); hook=None; ada=None
            try:
                env.noise=0;env.seed(es);env.reset();obs=env.set_init_state(init)
                for _ in range(SETTLING_STEPS): obs=env.step(DUMMY_ACTION)[0]
                if arm=='deep_block12': hook=ext.visual_encoder.backbone.blocks[11].register_forward_hook(lambda m,a,x:align(x,cal.ref(11),.75,'dead'))
                elif arm=='ours_v1': hook=ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda m,a,x:align(x,cal.ref(2),.75,'dead'))
                elif arm=='actmad_true': ada=ActMAD(policy,cal); ada.reset()
                gen=torch.Generator(device='cuda:0').manual_seed(ps); steps=0; success=False; finite=True
                with torch.no_grad():
                    while steps<a.max_steps:
                        raw=observed(obs,c['code'],ns,steps,c['dual'],wrapper); key=(steps,array_hash(env.sim.get_state().flatten()),*(array_hash(np.asarray(raw[k])) for k in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'))); raw=match_pixels(raw,key,cache,[]); po=policy_observation(raw,lang,policy)
                        if arm=='pixel_align': raw2=pixel_align_raw(po,cal)
                        else: raw2=po
                        if ada is not None:
                            from corp.preprocessing import prepare_observation; ada.update(prepare_observation(po,config=policy.config,normalizer=policy.normalizer,device='cuda:0'))
                        z=torch.randn((1,12,7),device='cuda:0',generator=gen); _,rep=visual_from_obs(policy,raw2); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu().numpy(); finite=finite and bool(np.isfinite(plan).all())
                        for ac in plan:
                            obs,rew,done,info=env.step(ac); steps+=1; success=bool(is_success(env,info,rew,done))
                            if done or success or steps>=a.max_steps: break
                        if done or success: break
                rows.append({'case_id':c['case_id'],'group':c['group'],'sub_group':c.get('sub_group'),'base_task':c['base_task'],'init_index':c['init_index'],'arm':arm,'success':success,'steps':steps,'finite':finite})
            finally:
                if hook: hook.remove()
                if ada: ada.close()
                env.close()
            print(json.dumps(rows[-1]),flush=True)
        tmp=dest.with_suffix('.pending');tmp.write_text(json.dumps(rows));tmp.rename(dest)
if __name__=='__main__': main()
