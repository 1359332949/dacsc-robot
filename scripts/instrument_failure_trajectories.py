"""Instrumentation-only failure-stage trajectories for P3 cases."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/hy-tmp/patch-diag')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'repos/CoRP')); sys.path.insert(0,str(ROOT/'repos/CoRP/scripts'))
from corp import load_policy  # noqa
from corp.representation.extractor import Representation  # noqa
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa
from target_only_action_validation import configure  # noqa
from dacsc.differentiable_policy import _encode_frozen, sample_from_z0  # noqa
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa
from scripts.run_photometric_policy import fit_params, apply_transform  # noqa


def parse_cases(s):
    out=[]
    for part in s.split(','):
        sid,dom,mode=part.split(':'); out.append((int(sid),dom,mode))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--photometric-fit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--cases',default='36:clean:frozen,37:clean:frozen,37:clean:transform,37:shift:frozen,37:shift:transform,43:shift:frozen,43:shift:transform,45:shift:frozen,45:shift:transform')
    ap.add_argument('--env-seed',type=int,default=7777); ap.add_argument('--z0-seed',type=int,default=7777)
    ap.add_argument('--executed-k',type=int,default=12); ap.add_argument('--max-steps',type=int,default=220)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args()
    cases=parse_cases(args.cases)
    for sid,dom,mode in cases:
        if dom not in ('clean','shift') or mode not in ('frozen','transform'): raise ValueError(f'bad case {sid}:{dom}:{mode}')
    if args.executed_k<=0 or args.max_steps<=0: raise ValueError('executed_k/max_steps must be positive')
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    split=load_split(args.split_json); validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id)
    variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    device=torch.device(args.device); policy=load_policy(args.checkpoint,device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    params={v:fit_params(args.photometric_fit,v) for v in ('agentview','eye_in_hand')}
    body_names=[str(x) for x in None] if False else None
    records=[]
    for state_id,domain,mode in cases:
        bddl=variant['env_bddl_name'] if domain=='shift' else variant['base_bddl']
        env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        try:
            np.random.seed(args.env_seed); torch.manual_seed(args.env_seed); env.seed(args.env_seed); env.reset(); obs=env.set_init_state(states[state_id])
            for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
            body_names=[str(x) for x in env.sim.model.body_names]
            steps=[]; step=0; chunk=0; success=False; done=False; clip=[]
            while not success and not done and step<args.max_steps:
                if mode=='transform':
                    use,cs=apply_transform(obs,params); clip.append(cs)
                else: use=obs
                po=policy_observation(use,language,policy)
                torch.manual_seed(args.z0_seed+100003*int(state_id)+chunk)
                z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
                pieces=_encode_frozen(policy,po)
                rep=Representation(tokens=torch.cat([pieces['visual_tokens'],pieces['language_tokens'].to(pieces['visual_tokens'].dtype),pieces['state_token']],dim=1),pooled_language=pieces['pooled_language'],action_dtype=pieces['visual_tokens'].dtype)
                with torch.no_grad(): ctrl=policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))
                if ctrl.shape[1]<args.executed_k: raise RuntimeError('action horizon < executed_k')
                for k in range(min(args.executed_k,ctrl.shape[1])):
                    raw=ctrl[0,k].detach().cpu().numpy().astype(float); clipped=np.clip(raw,-1,1)
                    obs,reward,done,info=env.step(clipped); step+=1
                    success=is_success(env,info,reward,done)
                    step_rec={'chunk':chunk,'k':k,'step':step,
                              'raw_action':raw.tolist(),'clipped_action':clipped.tolist(),
                              'post_ctrl':np.asarray(env.sim.data.ctrl,dtype=float).tolist(),
                              'eef_pos':np.asarray(obs.get('robot0_eef_pos',[]),dtype=float).tolist(),
                              'eef_quat':np.asarray(obs.get('robot0_eef_quat',[]),dtype=float).tolist(),
                              'gripper_qpos':np.asarray(obs.get('robot0_gripper_qpos',[]),dtype=float).tolist(),
                              'body_xpos':np.asarray(env.sim.data.body_xpos,dtype=float).tolist(),
                              'ncon':int(env.sim.data.ncon),'reward':float(reward),'success':bool(success),'done':bool(done)}
                    steps.append(step_rec)
                    if success or done or step>=args.max_steps: break
                chunk+=1
            records.append({'state_id':state_id,'domain':domain,'mode':mode,'success':bool(success),'steps':int(step),'chunk_count':chunk,
                            'body_names':body_names,'transform':params,
                            'transform_clip_mean':{v:float(np.mean([c[v] for c in clip])) if clip else None for v in ('agentview','eye_in_hand')},
                            'trajectory':steps})
            print(json.dumps({'state_id':state_id,'domain':domain,'mode':mode,'success':bool(success),'steps':int(step)}),flush=True)
        finally: env.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'failure_trajectory_v1','records':records},indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'n':len(records)}))

if __name__=='__main__': main()
