"""P0-A/P0-B continuous action intervention from same-observation Shift-Clean delta."""
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
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--state-ids',default='37,43,45'); ap.add_argument('--modes',default='baseline_clean,baseline_shift,p0b_chunk1,p0b_chunk2,fixed_chunk2')
    ap.add_argument('--env-seed',type=int,default=7777); ap.add_argument('--z0-seed',type=int,default=7777)
    ap.add_argument('--executed-k',type=int,default=12); ap.add_argument('--max-steps',type=int,default=220)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args()
    states_ids=[int(x) for x in args.state_ids.split(',') if x.strip()]; modes=[x.strip() for x in args.modes.split(',') if x.strip()]
    valid={'baseline_clean','baseline_shift','p0b_chunk1','p0b_chunk2','p0b_chunk3','fixed_chunk2'}
    if not set(modes)<=valid or not states_ids: raise ValueError('bad modes/states')
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
        for mode in modes:
            clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            try:
                np.random.seed(args.env_seed); torch.manual_seed(args.env_seed)
                if mode=='baseline_shift':
                    env=shift; env.seed(args.env_seed); env.reset(); obs=env.set_init_state(states[state_id])
                else:
                    env=clean; clean.seed(args.env_seed); clean.reset(); obs=clean.set_init_state(states[state_id]); shift.seed(args.env_seed); shift.reset()
                for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
                step=0; chunk=0; success=False; done=False; fixed_delta=None; injected=[]
                while not success and not done and step<args.max_steps:
                    po=policy_observation(obs,language,policy)
                    torch.manual_seed(args.z0_seed+100003*state_id+chunk)
                    z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
                    clean_plan=encode_observation(policy,po,z0) if mode!='baseline_shift' else None
                    if mode=='baseline_shift':
                        ctrl=encode_observation(policy,po,z0)['controller_actions']; delta=None
                    else:
                        ctrl=clean_plan['controller_actions']
                        if mode!='baseline_clean':
                            # synchronize Shift render at exact clean physical state
                            clean.sim.forward(); clean._update_observables(force=True); snap=snapshot(clean)
                            shift.seed(args.env_seed); shift.reset(); shift.env.sim.model.body_pos[:]=np.asarray(snap['model_signature']['body_pos'],float); shift.env.sim.model.body_quat[:]=np.asarray(snap['model_signature']['body_quat'],float); shift.env.sim.forward(); restore(shift,snap); shift._update_observables(force=True)
                            obs_s=shift.env._get_observations(); po_s=policy_observation(obs_s,language,policy); shift_plan=encode_observation(policy,po_s,z0)
                            delta=(shift_plan['controller_actions']-ctrl).detach().cpu().numpy()
                            if mode=='fixed_chunk2' and fixed_delta is None: fixed_delta=delta.copy()
                            if mode=='fixed_chunk2': delta=fixed_delta
                            n_inj=1 if mode=='p0b_chunk1' else (2 if mode in ('p0b_chunk2','fixed_chunk2') else 3)
                            inject_now=chunk<n_inj
                        else:
                            delta=None; inject_now=False
                    if ctrl.shape[1]<args.executed_k: raise RuntimeError('short action horizon')
                    for k in range(min(args.executed_k,ctrl.shape[1])):
                        act=ctrl[0,k].detach().cpu().numpy().astype(float)
                        if delta is not None and inject_now:
                            act=act+delta[0,k]
                            injected.append(float(np.max(np.abs(delta[0,k]))))
                        obs,reward,done,info=env.step(np.clip(act,-1,1)); step+=1
                        success=is_success(env,info,reward,done)
                        if success or done or step>=args.max_steps: break
                    chunk+=1
                target_pos=np.asarray(clean.sim.data.body_xpos[23],float) if mode!='baseline_shift' else None
                records.append({'state_id':state_id,'mode':mode,'success':bool(success),'steps':int(step),'injected_steps':len(injected),'injected_max':max(injected) if injected else None})
                print(json.dumps(records[-1]),flush=True)
            finally:
                clean.close(); shift.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'continuous_intervention_v1','records':records},indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'n':len(records)}))

if __name__=='__main__': main()
