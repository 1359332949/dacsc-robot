"""P0: early/late Clean-Shift phase swap at a fixed, predeclared step."""
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


def sync(src,dst):
    src.sim.forward(); src._update_observables(force=True)
    snap=snapshot(src); dst.seed(0); dst.reset()
    dst.env.sim.model.body_pos[:]=np.asarray(snap['model_signature']['body_pos'],float)
    dst.env.sim.model.body_quat[:]=np.asarray(snap['model_signature']['body_quat'],float)
    dst.env.sim.forward(); restore(dst,snap); dst._update_observables(force=True)
    return dst.env._get_observations(), compare(snap,snapshot(dst),tol={'body_xpos':1e-3,'eef_pos':1e-3,'body_xquat_rad':2e-3}), camera_compare(snap,snapshot(dst),expect_equal=True,tol=1e-3)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--state-ids',default='37,43,45'); ap.add_argument('--conditions',default='shift_to_clean,clean_to_shift')
    ap.add_argument('--switch-step',type=int,default=24); ap.add_argument('--env-seed',type=int,default=7777); ap.add_argument('--z0-seed',type=int,default=7777)
    ap.add_argument('--executed-k',type=int,default=12); ap.add_argument('--max-steps',type=int,default=220)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args()
    if args.switch_step<=0 or args.executed_k<=0 or args.max_steps<=0: raise ValueError('positive timing required')
    states_ids=[int(x) for x in args.state_ids.split(',') if x.strip()]; conds=[x.strip() for x in args.conditions.split(',') if x.strip()]
    if not set(conds)<={'shift_to_clean','clean_to_shift'}: raise ValueError('bad condition')
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
    for state_id in states_ids:
        for cond in conds:
            clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            envs={'clean':clean,'shift':shift}
            try:
                np.random.seed(args.env_seed); torch.manual_seed(args.env_seed)
                for e in envs.values(): e.seed(args.env_seed); e.reset()
                first='shift' if cond=='shift_to_clean' else 'clean'; second='clean' if first=='shift' else 'shift'
                active=first; env=envs[active]; obs=env.set_init_state(states[state_id])
                for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
                step=0; chunk=0; success=False; done=False; switched=False; switch_info=None; traj=[]
                while not success and not done and step<args.max_steps:
                    if not switched and step>=args.switch_step:
                        obs_new,comp,cam=sync(env,envs[second]); env=envs[second]; active=second; switched=True; obs=obs_new
                        switch_info={'from':first,'to':second,'step':step,'state_compare':bool(comp['all_pass']),'camera_compare':bool(cam['pass'])}; first=None
                    po=policy_observation(obs,language,policy)
                    torch.manual_seed(args.z0_seed+100003*state_id+chunk)
                    z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
                    ctrl=encode_observation(policy,po,z0)['controller_actions']
                    if ctrl.shape[1]<args.executed_k: raise RuntimeError('short horizon')
                    for k in range(min(args.executed_k,ctrl.shape[1])):
                        obs,reward,done,info=env.step(np.clip(ctrl[0,k].detach().cpu().numpy(),-1,1)); step+=1
                        success=is_success(env,info,reward,done); traj.append({'step':step,'active':active,'eef_pos':np.asarray(obs.get('robot0_eef_pos',[]),float).tolist(),'success':bool(success)})
                        if success or done or step>=args.max_steps: break
                    chunk+=1
                records.append({'state_id':state_id,'condition':cond,'first_domain':('shift' if cond=='shift_to_clean' else 'clean'),'second_domain':('clean' if cond=='shift_to_clean' else 'shift'),'switch_step':args.switch_step,'switch_info':switch_info,'success':bool(success),'steps':int(step),'trajectory':traj})
                print(json.dumps({'state_id':state_id,'condition':cond,'success':bool(success),'steps':step,'switch':switch_info}),flush=True)
            finally: clean.close(); shift.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'phase_swap_v1','records':records},indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'n':len(records)}))

if __name__=='__main__': main()
