"""T3-A/B/C: observation freshness, EEF site/body, same-model and cross-model restore."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/hy-tmp/patch-diag')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'repos/CoRP')); sys.path.insert(0, str(ROOT/'repos/CoRP/scripts'))
from corp import load_policy  # noqa
from corp.preprocessing import prepare_observation  # noqa
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation, state_from_observation  # noqa
from target_only_action_validation import configure  # noqa
from dacsc.state_pairing import snapshot, restore, digest, digest_model, digest_controller  # noqa
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa

def vec(a): return np.asarray(a, dtype=float).reshape(-1)
def qdist(q1,q2):
    q1=vec(q1); q2=vec(q2)
    return float(min(np.linalg.norm(q1-q2), np.linalg.norm(q1+q2)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--state-ids',default='0'); ap.add_argument('--seed',type=int,default=7777)
    ap.add_argument('--executed-k',type=int,default=12); ap.add_argument('--max-chunks',type=int,default=2)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    split=load_split(args.split_json); validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id)
    variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False)
    language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    ids=resolve_indices(split,'source_fit',args.state_ids)
    device=torch.device(args.device); policy=load_policy(args.checkpoint,device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    out={'schema':'corp_freshness_twin_audit_v1','records':[]}
    for state_id in ids:
        shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        cleanA=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        cleanB=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        try:
            def eef_body_id(env):
                return env.sim.model.body_name2id(env.robots[0].robot_model.eef_name)
            np.random.seed(args.seed); torch.manual_seed(args.seed)
            for e in (shift,cleanA,cleanB):
                e.seed(args.seed); e.reset(); e.set_init_state(states[state_id])
                for _ in range(SETTLING_STEPS): e.step(DUMMY_ACTION)
            obs_shift=shift.env._get_observations(); steps=0
            for chunk in range(args.max_chunks):
                # A: returned vs refreshed shift observation
                returned=shift.env._get_observations()
                q0=vec(shift.sim.data.qpos).copy(); v0=vec(shift.sim.data.qvel).copy(); t0=float(shift.sim.data.time)
                site0=vec(shift.sim.data.site_xpos[shift.robots[0].eef_site_id]).copy()
                body0=vec(shift.sim.data.body_xpos[eef_body_id(shift)]).copy()
                shift.sim.forward(); shift._update_observables(force=True)
                refreshed=shift.env._get_observations()
                q1=vec(shift.sim.data.qpos); v1=vec(shift.sim.data.qvel); t1=float(shift.sim.data.time)
                site1=vec(shift.sim.data.site_xpos[shift.robots[0].eef_site_id]); body1=vec(shift.sim.data.body_xpos[eef_body_id(shift)])
                raw_r=vec(state_from_observation(returned,policy.normalizer)); raw_f=vec(state_from_observation(refreshed,policy.normalizer))
                norm_r=policy.normalizer.normalize_state(torch.tensor(raw_r,dtype=torch.float32,device=device)[None]).detach().cpu().numpy()[0]
                norm_f=policy.normalizer.normalize_state(torch.tensor(raw_f,dtype=torch.float32,device=device)[None]).detach().cpu().numpy()[0]
                with torch.no_grad():
                    tok_r=policy.representation_extractor.state_encoder(torch.tensor(norm_r,dtype=torch.float32,device=device)[None]).detach().cpu().numpy()[0]
                    tok_f=policy.representation_extractor.state_encoder(torch.tensor(norm_f,dtype=torch.float32,device=device)[None]).detach().cpu().numpy()[0]
                rec_a={'state_id':int(state_id),'chunk':chunk,'step':int(steps),
                       'qpos_delta':float(np.max(np.abs(q0-q1))),'qvel_delta':float(np.max(np.abs(v0-v1))),
                       'time_delta':float(t1-t0),'site_delta':float(np.max(np.abs(site0-site1))),'body_delta':float(np.max(np.abs(body0-body1))),
                       'returned_vs_refreshed_raw_state_max':float(np.max(np.abs(raw_r-raw_f))),
                       'returned_vs_refreshed_norm_state_max':float(np.max(np.abs(norm_r-norm_f))),
                       'returned_vs_refreshed_state_token_l2':float(np.linalg.norm(tok_r-tok_f)),
                       'returned_eef_pos':vec(returned.get('robot0_eef_pos',[])).tolist(),
                       'refreshed_eef_pos':vec(refreshed.get('robot0_eef_pos',[])).tolist(),
                       'returned_eef_quat':vec(returned.get('robot0_eef_quat',[])).tolist(),
                       'refreshed_eef_quat':vec(refreshed.get('robot0_eef_quat',[])).tolist(),
                       'site_xpos':site1.tolist(),'body_xpos':body1.tolist()}
                # B/C: restore shift snapshot into cleanA (cross), then cleanA into cleanB (same model)
                snap=snapshot(shift)
                cleanA.seed(args.seed+991*chunk); cleanA.reset()
                cleanA.env.sim.model.body_pos[:]=np.asarray(snap['model_signature']['body_pos'],float)
                cleanA.env.sim.model.body_quat[:]=np.asarray(snap['model_signature']['body_quat'],float)
                cleanA.env.sim.forward(); restore(cleanA,snap); cleanA._update_observables(force=True)
                obsA=cleanA.env._get_observations(); snapA=snapshot(cleanA)
                rawA=vec(state_from_observation(obsA,policy.normalizer)); normA=policy.normalizer.normalize_state(torch.tensor(rawA,dtype=torch.float32,device=device)[None]).detach().cpu().numpy()[0]
                with torch.no_grad(): tokA=policy.representation_extractor.state_encoder(torch.tensor(normA,dtype=torch.float32,device=device)[None]).detach().cpu().numpy()[0]
                cleanB.seed(args.seed+1999*chunk); cleanB.reset()
                restore(cleanB,snapA); cleanB._update_observables(force=True)
                obsB=cleanB.env._get_observations(); snapB=snapshot(cleanB)
                rawB=vec(state_from_observation(obsB,policy.normalizer)); normB=policy.normalizer.normalize_state(torch.tensor(rawB,dtype=torch.float32,device=device)[None]).detach().cpu().numpy()[0]
                with torch.no_grad(): tokB=policy.representation_extractor.state_encoder(torch.tensor(normB,dtype=torch.float32,device=device)[None]).detach().cpu().numpy()[0]
                def pose(env,obs):
                    return {'site':vec(env.sim.data.site_xpos[env.robots[0].eef_site_id]),
                            'body':vec(env.sim.data.body_xpos[eef_body_id(env)]),
                            'obs_pos':vec(obs.get('robot0_eef_pos',[])),'obs_quat':vec(obs.get('robot0_eef_quat',[]))}
                p_shift=pose(shift,refreshed); pA=pose(cleanA,obsA); pB=pose(cleanB,obsB)
                rec_b={'state_id':int(state_id),'chunk':chunk,'step':int(steps),
                       'cross_site_delta':float(np.max(np.abs(p_shift['site']-pA['site']))),
                       'cross_body_delta':float(np.max(np.abs(p_shift['body']-pA['body']))),
                       'cross_obs_pos_delta':float(np.max(np.abs(p_shift['obs_pos']-pA['obs_pos']))),
                       'cross_obs_quat_angle':qdist(p_shift['obs_quat'],pA['obs_quat']),
                       'cross_raw_state_max':float(np.max(np.abs(raw_f-rawA))),
                       'cross_norm_state_max':float(np.max(np.abs(norm_f-normA))),
                       'cross_state_token_l2':float(np.linalg.norm(tok_f-tokA)),
                       'same_site_delta':float(np.max(np.abs(pA['site']-pB['site']))),
                       'same_body_delta':float(np.max(np.abs(pA['body']-pB['body']))),
                       'same_obs_pos_delta':float(np.max(np.abs(pA['obs_pos']-pB['obs_pos']))),
                       'same_obs_quat_angle':qdist(pA['obs_quat'],pB['obs_quat']),
                       'same_raw_state_max':float(np.max(np.abs(rawA-rawB))),
                       'same_norm_state_max':float(np.max(np.abs(normA-normB))),
                       'same_state_token_l2':float(np.linalg.norm(tokA-tokB)),
                       'site_minus_body_norm':float(np.linalg.norm(p_shift['site']-p_shift['body']))}
                out['records'].append({**rec_a,**rec_b})
                # execute actions on shift with same frozen policy actions
                po=policy_observation(shift.env._get_observations(),language,policy)
                torch.manual_seed(args.seed+100003*state_id+chunk)
                z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
                with torch.no_grad():
                    prep=prepare_observation(po,config=policy.config,normalizer=policy.normalizer,device=device)
                    # use policy.predict-like representation via policy_observation+extractor
                    enc=policy.representation_extractor(prep['images'],prep['instruction'],prep['state']) if False else None
                # use policy.predict for exact action
                act=policy.predict(observation=po,language=language,z0=z0) if False else None
                # fallback: use differentiable path
                from dacsc.differentiable_policy import _encode_frozen, sample_from_z0
                from corp.representation.extractor import Representation
                pieces=_encode_frozen(policy,po)
                rep=Representation(tokens=torch.cat([pieces['visual_tokens'],pieces['language_tokens'].to(pieces['visual_tokens'].dtype),pieces['state_token']],dim=1),pooled_language=pieces['pooled_language'],action_dtype=pieces['visual_tokens'].dtype)
                with torch.no_grad(): ctrl=policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))
                for k in range(min(args.executed_k,ctrl.shape[1])):
                    obs_shift,r,d,_=shift.step(np.clip(ctrl[0,k].cpu().numpy(),-1,1)); steps+=1
                    if d: break
                if d: break
            print(json.dumps({'state_id':int(state_id),'records':len(out['records'])}),flush=True)
        finally:
            shift.close(); cleanA.close(); cleanB.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'out':str(args.out),'n':len(out['records'])}))

if __name__=='__main__': main()
