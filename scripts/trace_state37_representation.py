"""Experiment A: fixed-physical-state Clean / Clean+Transform / Shift tracing."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/hy-tmp/patch-diag')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'repos/CoRP')); sys.path.insert(0,str(ROOT/'repos/CoRP/scripts'))
from corp import load_policy  # noqa
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa
from target_only_action_validation import configure  # noqa
from dacsc.state_pairing import snapshot, restore, compare, camera_compare, model_compatible  # noqa
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa
from scripts.run_photometric_policy import fit_params, apply_transform  # noqa


def rel(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    return float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-8))


def quat_angle(q1,q2):
    q1=np.asarray(q1,dtype=float); q2=np.asarray(q2,dtype=float)
    q1=q1/(np.linalg.norm(q1)+1e-8); q2=q2/(np.linalg.norm(q2)+1e-8)
    return float(2*np.arccos(np.clip(abs(np.dot(q1,q2)),-1,1)))


def target_object_from_bddl(path):
    text=Path(path).read_text(); m=re.search(r'\(:goal\s*\(And\s*\(On\s+([^\s\)]+)',text,re.S)
    if not m: raise RuntimeError('cannot parse target object')
    return m.group(1)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--photometric-fit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--state-id',type=int,default=37)
    ap.add_argument('--env-seed',type=int,default=7777); ap.add_argument('--z0-seed',type=int,default=7777)
    ap.add_argument('--max-steps',type=int,default=24); ap.add_argument('--executed-k',type=int,default=12)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args()
    if args.max_steps<=0 or args.executed_k<=0: raise ValueError('positive steps required')
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
    clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
    shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
    try:
        np.random.seed(args.env_seed); torch.manual_seed(args.env_seed)
        clean.seed(args.env_seed); clean.reset(); obs=clean.set_init_state(states[args.state_id])
        for _ in range(SETTLING_STEPS): obs,_,_,_=clean.step(DUMMY_ACTION)
        shift.seed(args.env_seed); shift.reset()
        body_names=[str(x) for x in clean.sim.model.body_names]
        target_name=target_object_from_bddl(variant['base_bddl'])
        target_bodies=[i for i,n in enumerate(body_names) if n==f'{target_name}_main']
        if len(target_bodies)!=1: raise RuntimeError(f'target body not unique: {target_name} {target_bodies}')
        target_body_id=target_bodies[0]
        geom_names=[str(x) for x in clean.sim.model.geom_names]
        target_geoms=[i for i,n in enumerate(geom_names) if int(clean.sim.model.geom_bodyid[i])==target_body_id]
        finger_geoms=[i for i,n in enumerate(geom_names) if 'finger' in n and ('gripper0' in n or 'robot0' in n)]
        if not target_geoms or not finger_geoms: raise RuntimeError('target/finger geom resolution failed')
        target_geometry={'target_object':target_name,'target_body':f'{target_name}_main','target_body_id':target_body_id,
                         'target_geoms':[geom_names[i] for i in target_geoms],'finger_geoms':[geom_names[i] for i in finger_geoms],
                         'target_initial_pos':np.asarray(clean.sim.data.body_xpos[target_body_id],dtype=float).tolist(),
                         'note':'target body origin is not necessarily the grasp point; see EEF-target distance'}
        records=[]; step=0; chunk=0
        while step<args.max_steps:
            # sync shift env to the exact clean physical state before encoding
            clean.sim.forward(); clean._update_observables(force=True)
            snap=snapshot(clean)
            shift.seed(args.env_seed); shift.reset()
            shift.env.sim.model.body_pos[:]=np.asarray(snap['model_signature']['body_pos'],float)
            shift.env.sim.model.body_quat[:]=np.asarray(snap['model_signature']['body_quat'],float)
            shift.env.sim.forward(); restore(shift,snap); shift._update_observables(force=True)
            obs_shift=shift.env._get_observations()
            compat=compare(snap,snapshot(shift),tol={'body_xpos':1e-3,'eef_pos':1e-3,'body_xquat_rad':2e-3})
            cam=camera_compare(snap,snapshot(shift),expect_equal=True,tol=1e-3)
            # chunk-level raw action plan
            po=policy_observation(obs,language,policy)
            torch.manual_seed(args.z0_seed+100003*args.state_id+chunk)
            z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
            raw_plan=encode_observation(policy,po,z0)
            for k in range(min(args.executed_k,raw_plan['controller_actions'].shape[1])):
                po_c=policy_observation(obs,language,policy)
                use_t,clip_stats=apply_transform(obs,params); po_t=policy_observation(use_t,language,policy); po_s=policy_observation(obs_shift,language,policy)
                enc_c=encode_observation(policy,po_c,z0); enc_t=encode_observation(policy,po_t,z0); enc_s=encode_observation(policy,po_s,z0)
                dino={}
                for v in enc_c['patches']:
                    pc=enc_c['patches'][v].float().cpu().numpy(); pt=enc_t['patches'][v].float().cpu().numpy(); ps=enc_s['patches'][v].float().cpu().numpy()
                    def cosine(a,b):
                        aa=a.reshape(-1,a.shape[-1]); bb=b.reshape(-1,b.shape[-1])
                        return float(np.mean(np.sum(aa*bb,axis=-1)/(np.linalg.norm(aa,axis=-1)*np.linalg.norm(bb,axis=-1)+1e-8)))
                    def patch_stats(a,b):
                        diff=np.linalg.norm((a-b).reshape(-1,a.shape[-1]),axis=-1)/(np.linalg.norm(a.reshape(-1,a.shape[-1]),axis=-1)+1e-8)
                        return {'patch_l2_mean':float(diff.mean()),'patch_l2_max':float(diff.max()),
                                'patch_l2_top10_mean':float(np.sort(diff)[-10:].mean())}
                    dino[v]={'clean_vs_transform':rel(pc,pt),'clean_vs_shift':rel(pc,ps),
                             'clean_vs_transform_cosine':cosine(pc,pt),'clean_vs_shift_cosine':cosine(pc,ps),
                             'clean_vs_transform_patch':patch_stats(pc,pt),'clean_vs_shift_patch':patch_stats(pc,ps)}
                ac=enc_c['controller_actions'][0,0].detach().cpu().numpy(); at=enc_t['controller_actions'][0,0].detach().cpu().numpy(); ash=enc_s['controller_actions'][0,0].detach().cpu().numpy()
                def adiff(a,b):
                    def one(x,y):
                        d=y-x; return {'xyz':float(np.sqrt(np.mean(d[:3]**2))),'rot':float(np.sqrt(np.mean(d[3:6]**2))),'grip':float(abs(d[6])),'norm':float(np.sqrt(np.mean(d**2)))}
                    return {'pre':one(a,b),'post':one(np.clip(a,-1,1),np.clip(b,-1,1))}
                eef=np.asarray(obs.get('robot0_eef_pos',[]),float); tgt=np.asarray(clean.sim.data.body_xpos[target_body_id],float)
                eef_q=np.asarray(obs.get('robot0_eef_quat',[]),float); tgt_q=np.asarray(clean.sim.data.body_xquat[target_body_id],float)
                target_contact=False
                for ci in range(int(clean.sim.data.ncon)):
                    c=clean.sim.data.contact[ci]
                    if ((int(c.geom1) in target_geoms and int(c.geom2) in finger_geoms) or
                        (int(c.geom2) in target_geoms and int(c.geom1) in finger_geoms)):
                        target_contact=True; break
                records.append({'state_id':args.state_id,'step':step,'chunk':chunk,'k':k,
                                'dino':dino,'resampler':{'clean_vs_transform':rel(enc_c['visual_tokens'].float().cpu().numpy(),enc_t['visual_tokens'].float().cpu().numpy()),
                                                           'clean_vs_shift':rel(enc_c['visual_tokens'].float().cpu().numpy(),enc_s['visual_tokens'].float().cpu().numpy())},
                                'action':{'clean_vs_transform':adiff(ac,at),'clean_vs_shift':adiff(ac,ash)},
                                'eef_pos':eef.tolist(),'target_pos':tgt.tolist(),'eef_target_distance':float(np.linalg.norm(eef-tgt)),
                                'eef_quat':eef_q.tolist(),'target_quat':tgt_q.tolist(),'theta_rel':quat_angle(eef_q,tgt_q),'eef_to_target_body_origin':float(np.linalg.norm(eef-tgt)),
                                'target_finger_contact':bool(target_contact),'pairing_pass':bool(compat['all_pass'] and cam['pass']),
                                'clip_fraction':clip_stats})
                obs,reward,done,info=clean.step(np.clip(raw_plan['controller_actions'][0,k].detach().cpu().numpy(),-1,1)); step+=1
                if done or step>=args.max_steps: break
            chunk+=1
        args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'state37_three_input_trace_v1','state_id':args.state_id,'transform':params,'target_geometry':target_geometry,'records':records},indent=2)+'\n')
        print(json.dumps({'out':str(args.out),'n':len(records),'target':target_geometry['target_object'],'target_geoms':len(target_geometry['target_geoms']),'finger_geoms':len(target_geometry['finger_geoms'])},indent=2))
    finally:
        clean.close(); shift.close()

if __name__=='__main__': main()
