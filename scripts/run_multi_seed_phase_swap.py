"""Multi-seed phase-swap + close-time OBB geometry."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/hy-tmp/patch-diag')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'repos/CoRP')); sys.path.insert(0,str(ROOT/'repos/CoRP/scripts'))
from corp import load_policy  # noqa
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa
from target_only_action_validation import configure  # noqa
from dacsc.state_pairing import snapshot, restore, compare, camera_compare  # noqa
from dacsc.tta_protocol import load_split, validate_split_identity  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa

TARGET_BODY='akita_black_bowl_1_main'


def sync(src,dst,seed):
    src.sim.forward(); src._update_observables(force=True); snap=snapshot(src)
    dst.seed(seed); dst.reset()
    dst.env.sim.model.body_pos[:]=np.asarray(snap['model_signature']['body_pos'],float)
    dst.env.sim.model.body_quat[:]=np.asarray(snap['model_signature']['body_quat'],float)
    dst.env.sim.forward(); restore(dst,snap); dst._update_observables(force=True)
    return dst.env._get_observations(), compare(snap,snapshot(dst),tol={'body_xpos':1e-3,'eef_pos':1e-3,'body_xquat_rad':2e-3}), camera_compare(snap,snapshot(dst),expect_equal=True,tol=1e-3)


def obb_ids(env):
    body_names=[str(x) for x in env.sim.model.body_names]; geom_names=[str(x) for x in env.sim.model.geom_names]
    if TARGET_BODY not in body_names: raise RuntimeError('target body missing')
    tid=body_names.index(TARGET_BODY)
    boxes=[i for i in range(len(geom_names)) if int(env.sim.model.geom_bodyid[i])==tid and int(env.sim.model.geom_type[i])==6]
    pads=[i for i,n in enumerate(geom_names) if 'finger' in n and 'pad_collision' in n]
    if len(boxes)!=40 or len(pads)!=2: raise RuntimeError(f'geometry count boxes={len(boxes)} pads={len(pads)}')
    return boxes,pads


def obb_metrics(env,ids):
    boxes,pads=ids
    res=[]
    for pi in pads:
        p=np.asarray(env.sim.data.geom_xpos[pi],float); best=None
        for bi in boxes:
            c=np.asarray(env.sim.data.geom_xpos[bi],float); R=np.asarray(env.sim.data.geom_xmat[bi],float).reshape(3,3); size=np.asarray(env.sim.model.geom_size[bi],float)
            loc=R.T@(p-c); q=np.clip(loc,-size,size); delta=loc-q; dist=float(np.linalg.norm(delta))
            if best is None or dist<best[0]:
                nloc=delta/(np.linalg.norm(delta)+1e-12) if np.linalg.norm(delta)>1e-12 else np.zeros(3)
                best=(dist,(R@nloc).tolist())
        res.append({'pad':str(env.sim.model.geom_names[pi]),'distance':best[0],'normal':best[1]})
    pc=[np.asarray(env.sim.data.geom_xpos[i],float) for i in pads]; axis=pc[1]-pc[0]; axis=axis/(np.linalg.norm(axis)+1e-12)
    ang=[]
    for r in res:
        n=np.asarray(r['normal'],float); ang.append(float(np.arccos(np.clip(abs(np.dot(axis,n)),-1,1))))
    return {'pads':res,'closing_axis':axis.tolist(),'angles_rad':ang}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--state-ids',default='43,45'); ap.add_argument('--conditions',default='baseline_clean,baseline_shift,shift_to_clean,clean_to_shift')
    ap.add_argument('--seeds',default='7777,8888,9999'); ap.add_argument('--switch-step',type=int,default=24)
    ap.add_argument('--executed-k',type=int,default=12); ap.add_argument('--max-steps',type=int,default=220)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args()
    ids=[int(x) for x in args.state_ids.split(',') if x.strip()]; conds=[x.strip() for x in args.conditions.split(',') if x.strip()]; seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    valid={'baseline_clean','baseline_shift','shift_to_clean','clean_to_shift'}
    if not set(conds)<=valid or not ids or not seeds: raise ValueError('bad args')
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    split=load_split(args.split_json); validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id)
    variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    device=torch.device(args.device); policy=load_policy(args.checkpoint,device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    records=[]
    for state_id in ids:
        for cond in conds:
            first=('shift' if cond in ('baseline_shift','shift_to_clean') else 'clean')
            second=None if cond.startswith('baseline') else ('clean' if first=='shift' else 'shift')
            for seed in seeds:
                clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
                shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
                envs={'clean':clean,'shift':shift}
                try:
                    np.random.seed(seed); torch.manual_seed(seed)
                    for e in envs.values(): e.seed(seed); e.reset()
                    env=envs[first]; obs=env.set_init_state(states[state_id])
                    for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
                    ids_geom=obb_ids(env); step=0; chunk=0; success=False; done=False; switched=False; switch_info=None
                    close_geom=None; min_left=1e9; min_right=1e9
                    while not success and not done and step<args.max_steps:
                        if second is not None and not switched and step>=args.switch_step:
                            obs,comp,cam=sync(env,envs[second],seed); env=envs[second]; switched=True
                            switch_info={'to':second,'step':step,'state_compare':bool(comp['all_pass']),'camera_compare':bool(cam['pass'])}; obs_hold=obs
                        po=policy_observation(obs,language,policy)
                        torch.manual_seed(seed+100003*state_id+chunk)
                        z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
                        ctrl=encode_observation(policy,po,z0)['controller_actions']
                        if ctrl.shape[1]<args.executed_k: raise RuntimeError('short horizon')
                        for k in range(min(args.executed_k,ctrl.shape[1])):
                            act=np.clip(ctrl[0,k].detach().cpu().numpy(),-1,1)
                            obs,reward,done,info=env.step(act); step+=1
                            success=is_success(env,info,reward,done)
                            if act[6]>0 and close_geom is None: close_geom=obb_metrics(env,ids_geom)
                            if step%4==0 or success or done:
                                m=obb_metrics(env,ids_geom); min_left=min(min_left,m['pads'][0]['distance']); min_right=min(min_right,m['pads'][1]['distance'])
                            if success or done or step>=args.max_steps: break
                        chunk+=1
                    records.append({'state_id':state_id,'condition':cond,'seed':seed,'success':bool(success),'steps':int(step),'switch':switch_info,
                                    'close_obb':close_geom,'min_left_pad_distance':None if min_left>1e8 else float(min_left),'min_right_pad_distance':None if min_right>1e8 else float(min_right)})
                    print(json.dumps({'state':state_id,'cond':cond,'seed':seed,'succ':bool(success),'steps':step}),flush=True)
                finally: clean.close(); shift.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'multi_seed_phase_swap_v1','records':records},indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'n':len(records)}))

if __name__=='__main__': main()
