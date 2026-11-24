"""True 2x2 checkpoint fork at step 24 for Clean/Shift suffixes."""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/hy-tmp/patch-diag')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'repos/CoRP')); sys.path.insert(0,str(ROOT/'repos/CoRP/scripts'))
from corp import load_policy  # noqa
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa
from target_only_action_validation import configure  # noqa
from dacsc.tta_protocol import load_split, validate_split_identity  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa


def prefix_run(env,policy,language,state_id,seed,steps,chunk_steps=12):
    np.random.seed(seed); torch.manual_seed(seed); env.seed(seed); env.reset(); obs=env.set_init_state(states[state_id])
    for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
    done=False; chunk=0; step=0
    while step<steps and not done:
        po=policy_observation(obs,language,policy)
        torch.manual_seed(seed+100003*state_id+chunk)
        z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
        ctrl=encode_observation(policy,po,z0)['controller_actions']
        for k in range(min(chunk_steps,ctrl.shape[1])):
            obs,reward,done,info=env.step(np.clip(ctrl[0,k].detach().cpu().numpy(),-1,1)); step+=1
            if step>=steps or done: break
        chunk+=1
    env.sim.forward(); env._update_observables(force=True)
    ctrl_state={}
    for k,v in env.robots[0].controller.__dict__.items():
        try: ctrl_state[k]=copy.deepcopy(v)
        except Exception: pass
    snap={'sim':env.sim.get_state(),'controller':ctrl_state}
    return snap,step


def branch_run(env,snap,policy,language,state_id,seed,max_steps,start_step=24,chunk_steps=12):
    env.seed(seed); env.reset(); env.sim.set_state(snap['sim'])
    for k,v in snap['controller'].items():
        try: setattr(env.robots[0].controller,k,copy.deepcopy(v))
        except Exception: pass
    env.sim.forward(); env._update_observables(force=True)
    obs=env.env._get_observations(); step=start_step; chunk=start_step//chunk_steps; success=False; done=False
    while step<max_steps and not success and not done:
        po=policy_observation(obs,language,policy)
        torch.manual_seed(seed+100003*state_id+chunk)
        z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
        ctrl=encode_observation(policy,po,z0)['controller_actions']
        for k in range(min(chunk_steps,ctrl.shape[1])):
            obs,reward,done,info=env.step(np.clip(ctrl[0,k].detach().cpu().numpy(),-1,1)); step+=1
            success=is_success(env,info,reward,done)
            if success or done or step>=max_steps: break
        chunk+=1
    return success,step,chunk


def main():
    global states,device
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--state-ids',default='43,45'); ap.add_argument('--seeds',default='7777,8888,9999')
    ap.add_argument('--prefix-steps',type=int,default=24); ap.add_argument('--max-steps',type=int,default=220)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args()
    ids=[int(x) for x in args.state_ids.split(',') if x.strip()]; seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    if args.prefix_steps%12!=0 or args.prefix_steps<=0: raise ValueError('prefix_steps must be multiples of 12')
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
        for seed in seeds:
            clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            try:
                csnap,cstep=prefix_run(clean,policy,language,state_id,seed,args.prefix_steps)
                ssnap,sstep=prefix_run(shift,policy,language,state_id,seed,args.prefix_steps)
                # C-prefix branches
                for suffix,env in (('clean',clean),('shift',shift)):
                    ok,st,ch=branch_run(env,csnap,policy,language,state_id,seed,args.max_steps,args.prefix_steps)
                    records.append({'state_id':state_id,'seed':seed,'prefix':'clean','suffix':suffix,'success':bool(ok),'steps':st,'prefix_steps':cstep})
                    print(json.dumps(records[-1]),flush=True)
                # S-prefix branches
                for suffix,env in (('clean',clean),('shift',shift)):
                    ok,st,ch=branch_run(env,ssnap,policy,language,state_id,seed,args.max_steps,args.prefix_steps)
                    records.append({'state_id':state_id,'seed':seed,'prefix':'shift','suffix':suffix,'success':bool(ok),'steps':st,'prefix_steps':sstep})
                    print(json.dumps(records[-1]),flush=True)
            finally: clean.close(); shift.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'checkpoint_fork_v1','records':records},indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'n':len(records)}))

if __name__=='__main__': main()
