"""Causal action intervention: inject early Transform action delta into Frozen."""
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
from dacsc.differentiable_policy import _encode_frozen, sample_from_z0  # noqa
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa
from scripts.run_photometric_policy import fit_params, apply_transform  # noqa


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--photometric-fit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--state-id',type=int,default=37)
    ap.add_argument('--domains',default='clean,shift'); ap.add_argument('--modes',default='baseline,one_step,xyz_one_step')
    ap.add_argument('--env-seed',type=int,default=7777); ap.add_argument('--z0-seed',type=int,default=7777)
    ap.add_argument('--executed-k',type=int,default=12); ap.add_argument('--max-steps',type=int,default=220)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args()
    domains=[x.strip() for x in args.domains.split(',') if x.strip()]; modes=[x.strip() for x in args.modes.split(',') if x.strip()]
    if not domains or not set(domains)<={'clean','shift'}: raise ValueError('bad domains')
    if not modes or not set(modes)<={'baseline','one_step','xyz_one_step'}: raise ValueError('bad modes')
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
    records=[]
    for domain in domains:
        bddl=variant['env_bddl_name'] if domain=='shift' else variant['base_bddl']
        for mode in modes:
            env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            try:
                np.random.seed(args.env_seed); torch.manual_seed(args.env_seed); env.seed(args.env_seed); env.reset(); obs=env.set_init_state(states[args.state_id])
                for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
                step=0; chunk=0; success=False; done=False; injected=None; steps=[]
                while not success and not done and step<args.max_steps:
                    po=policy_observation(obs,language,policy)
                    torch.manual_seed(args.z0_seed+100003*args.state_id+chunk)
                    z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
                    pieces=_encode_frozen(policy,po)
                    from corp.representation.extractor import Representation
                    rep=Representation(tokens=torch.cat([pieces['visual_tokens'],pieces['language_tokens'].to(pieces['visual_tokens'].dtype),pieces['state_token']],dim=1),pooled_language=pieces['pooled_language'],action_dtype=pieces['visual_tokens'].dtype)
                    with torch.no_grad(): ctrl=policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))
                    if ctrl.shape[1]<args.executed_k: raise RuntimeError('action horizon too short')
                    # one-step injection at the first executed action of chunk 0
                    if chunk==0 and mode!='baseline':
                        use_t,_=apply_transform(obs,params); po_t=policy_observation(use_t,language,policy); pieces_t=_encode_frozen(policy,po_t)
                        rep_t=Representation(tokens=torch.cat([pieces_t['visual_tokens'],pieces_t['language_tokens'].to(pieces_t['visual_tokens'].dtype),pieces_t['state_token']],dim=1),pooled_language=pieces_t['pooled_language'],action_dtype=pieces_t['visual_tokens'].dtype)
                        with torch.no_grad(): ctrl_t=policy.normalizer.unnormalize_action(sample_from_z0(policy,rep_t,z0))
                        delta=(ctrl_t[0,0]-ctrl[0,0]).detach().cpu().numpy()
                        if mode=='xyz_one_step': delta[3:]=0
                        injected={'delta':delta.tolist(),'max_abs':float(np.max(np.abs(delta)))}
                    for k in range(min(args.executed_k,ctrl.shape[1])):
                        act=ctrl[0,k].detach().cpu().numpy().astype(float)
                        if chunk==0 and k==0 and injected is not None: act=act+np.asarray(injected['delta'],float)
                        obs,reward,done,info=env.step(np.clip(act,-1,1)); step+=1
                        success=is_success(env,info,reward,done)
                        steps.append({'step':step,'action':np.clip(act,-1,1).tolist(),'eef_pos':np.asarray(obs.get('robot0_eef_pos',[]),dtype=float).tolist(),'gripper_qpos':np.asarray(obs.get('robot0_gripper_qpos',[]),dtype=float).tolist(),'success':bool(success)})
                        if success or done or step>=args.max_steps: break
                    chunk+=1
                records.append({'state_id':args.state_id,'domain':domain,'mode':mode,'success':bool(success),'steps':int(step),'injected':injected,'trajectory':steps})
                print(json.dumps({'domain':domain,'mode':mode,'success':bool(success),'steps':step,'injected_max':injected['max_abs'] if injected else None}),flush=True)
            finally: env.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'action_intervention_v1','params':params,'records':records},indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'n':len(records)}))

if __name__=='__main__': main()
