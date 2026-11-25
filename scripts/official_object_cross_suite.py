"""Official LIBERO-Object physical cross-suite: build manifest and 3-arm EGL rollouts."""
from __future__ import annotations
import argparse,ctypes,hashlib,json,os,random,sys
from pathlib import Path
import numpy as np
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
from dacsc.static_visual_v2 import array_hash,file_hash,seed_for,sensor_transform
from dacsc.single_block_probe import align,VIEWS
from egl_shallow_block3_closed_loop import match_pixels,observed
TASKS=['pick_up_the_alphabet_soup_and_place_it_in_the_basket','pick_up_the_bbq_sauce_and_place_it_in_the_basket','pick_up_the_butter_and_place_it_in_the_basket','pick_up_the_chocolate_pudding_and_place_it_in_the_basket','pick_up_the_cream_cheese_and_place_it_in_the_basket','pick_up_the_ketchup_and_place_it_in_the_basket','pick_up_the_milk_and_place_it_in_the_basket','pick_up_the_orange_juice_and_place_it_in_the_basket','pick_up_the_salad_dressing_and_place_it_in_the_basket','pick_up_the_tomato_sauce_and_place_it_in_the_basket']
TAG='official_object_physical_v1'
ARMS=('frozen','spatial_ref','object_clean_cal')

def build(out):
    cls=json.load(open(LIB/'libero/libero/benchmark/task_classification.json'))['libero_object']
    bycat={}
    for r in cls: bycat.setdefault(r['category'],[]).append(r)
    cases=[]
    def add(task,factor,sub,bddl,init,idx,noise=0,dual=False,tag=None):
        c=dict(case_id=hashlib.sha256(json.dumps([TAG,task,factor,sub,str(bddl),idx,noise,dual]).encode()).hexdigest()[:24],task=task,factor=factor,sub=sub,bddl=str(bddl.relative_to(LIB)) if str(bddl).startswith(str(LIB)) else str(bddl),init_file=str(init.relative_to(LIB)),init_index=idx,noise=noise,dual=dual,env_seed=seed_for(TAG,task,factor,sub,idx,'env'),sensor_seed=seed_for(TAG,task,factor,sub,'sensor'),policy_seed=seed_for(TAG,task,factor,sub,'policy'))
        cases.append(c)
    for task in TASKS:
        base=LIB/'libero/libero/bddl_files/libero_object'/f'{task}.bddl'; init=LIB/'libero/libero/init_files/libero_object'/f'{task}.pruned_init'
        assert base.exists() and init.exists(),task
        add(task,'clean','clean',base,init,0)
        for cat,fac,sub in [('Light Conditions','lighting','light_asset'),('Background Textures','background','background_texture'),('Sensor Noise','sensor_noise','sensor_noise')]:
            cand=[r for r in bycat[cat] if r['name'].startswith(task+'_')]
            assert len(cand)>=3,(task,cat,len(cand))
            sel=sorted(cand,key=lambda r:seed_for(TAG,task,fac,r['name']))[:3]
            for r in sel:
                nm=r['name']
                if fac=='sensor_noise':
                    import re; m=re.search(r'_noise_(\d+)$',nm); assert m,nm; code=int(m.group(1)); add(task,fac,sub,base,init,0,noise=code)
                    add(task,fac,sub+'_bilateral',base,init,0,noise=code,dual=True)
                else:
                    b=LIB/'libero/libero/bddl_files/libero_object'/f'{nm}.bddl'; assert b.exists(),b; add(task,fac,sub,b,init,0)
    assert len(cases)==130,len(cases)
    out.write_text(json.dumps({'tag':TAG,'suite':'libero_object','cases':cases},indent=2)+'\n')
    print('wrote',out,len(cases))

def run(a):
    import torch
    from corp import load_policy
    from target_only_action_validation import sample_z,visual_from_obs
    from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.benchmark import grab_language_from_filename
    import libero.libero.envs.env_wrapper as wrapper
    assert os.environ['MUJOCO_GL']==os.environ['PYOPENGL_PLATFORM']=='egl'
    m=json.load(open(a.manifest)); assert m['tag']==TAG and len(m['cases'])==130
    sp=np.load(ROOT/'runs/object_transfer_synthetic_v1/spatial_transfer.npz',allow_pickle=False); spatial=tuple(torch.as_tensor(sp[k],device='cuda:0') for k in ('mean','std','tau'))
    oz=np.load(ROOT/'runs/object_transfer_synthetic_v1/object_loto.npz',allow_pickle=False); object_refs={t:tuple(torch.as_tensor(oz[f'{t}__3__{k}'],device='cuda:0') for k in ('mean','std','tau')) for t in TASKS}
    policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False); ext=policy.representation_extractor; assert len(ext.visual_encoder.backbone.blocks)==12
    jobs=[(c,s) for c in m['cases'] for s in (0,1,2) if c['case_id'] in {x['case_id'] for x in m['cases']}]
    for c,s in jobs[a.shard::a.workers]:
        dest=a.out/f"{'dual' if c['dual'] else 'std'}_{c['case_id']}_{s}.json"
        if dest.exists(): continue
        init=torch.load(LIB/c['init_file'],weights_only=False,map_location='cpu')[c['init_index']]; lang=grab_language_from_filename('libero_object',c['task']+'.bddl'); cache={}; rows=[]
        for arm in ARMS:
            es=seed_for(TAG,c['task'],c['factor'],c['sub'],c['init_index'],'env',s); ns=seed_for(TAG,c['task'],c['factor'],c['sub'],'sensor',s); ps=seed_for(TAG,c['task'],c['factor'],c['sub'],'policy',s)
            random.seed(es); np.random.seed(es); torch.manual_seed(es); ctypes.CDLL(None).srand(ctypes.c_uint(es))
            env=OffScreenRenderEnv(bddl_file_name=str(LIB/c['bddl']),camera_heights=256,camera_widths=256,render_gpu_device_id=0); hook=None
            try:
                env.noise=0; env.seed(es); env.reset(); obs=env.set_init_state(init)
                for _ in range(SETTLING_STEPS): obs=env.step(DUMMY_ACTION)[0]
                if arm!='frozen':
                    ref=spatial if arm=='spatial_ref' else object_refs[c['task']]
                    hook=ext.visual_encoder.backbone.blocks[2].register_forward_hook(lambda mod,args,x:align(x,ref,.75,'dead'))
                g=torch.Generator(device='cuda:0').manual_seed(ps); steps=0; success=False
                with torch.no_grad():
                    while steps<220:
                        raw=observed(obs,c['noise'],ns,steps,c['dual'],wrapper); key=(steps,array_hash(env.sim.get_state().flatten()),*(array_hash(np.asarray(raw[k])) for k in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'))); raw=match_pixels(raw,key,cache,[]); po=policy_observation(raw,lang,policy); z=torch.randn((1,12,7),device='cuda:0',generator=g); _,rep=visual_from_obs(policy,po); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu().numpy()
                        for ac in plan:
                            obs,rew,done,info=env.step(ac); steps+=1; success=bool(is_success(env,info,rew,done))
                            if done or success or steps>=220: break
                        if done or success: break
                rows.append({'arm':arm,'success':success,'steps':steps,'case_id':c['case_id'],'task':c['task'],'factor':c['factor'],'sub':c['sub'],'dual':c['dual'],'seed':s})
                print(json.dumps(rows[-1]),flush=True)
            finally:
                if hook: hook.remove()
                env.close()
        tmp=dest.with_suffix('.pending'); tmp.write_text(json.dumps(rows,allow_nan=False)); tmp.rename(dest)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('build','run')); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--manifest',type=Path); ap.add_argument('--workers',type=int,default=16); ap.add_argument('--shard',type=int,default=0); a=ap.parse_args()
    (build(a.out) if a.mode=='build' else run(a))
