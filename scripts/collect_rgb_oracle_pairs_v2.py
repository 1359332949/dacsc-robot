"""Collect one real RGB Clean/Shift pair per source state for DINO oracle."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch

ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from corp import load_policy  # noqa
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation, state_from_observation  # noqa
from target_only_action_validation import configure  # noqa
from dacsc.state_pairing import snapshot, restore, compare, camera_compare  # noqa
from dacsc.tta_protocol import load_split, validate_split_identity  # noqa


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--env-seed',type=int,default=7777); ap.add_argument('--z0-seed',type=int,default=7777); ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    split=load_split(args.split_json); matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id); variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    device=torch.device(args.device); policy=load_policy(args.checkpoint,device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    rows=[]; meta=[]
    for role,ids in (('source_fit',split['source_train_state_ids']),('source_validation',split['source_val_state_ids'])):
        for sid in ids:
            clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu); shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            try:
                clean.seed(args.env_seed); clean.reset(); obs=clean.set_init_state(states[sid]); shift.seed(args.env_seed); shift.reset()
                for _ in range(SETTLING_STEPS): obs,_,_,_=clean.step(DUMMY_ACTION)
                clean.sim.forward(); clean._update_observables(force=True); snap=snapshot(clean)
                shift.seed(args.env_seed); shift.reset(); shift.env.sim.model.body_pos[:]=np.asarray(snap['model_signature']['body_pos'],float); shift.env.sim.model.body_quat[:]=np.asarray(snap['model_signature']['body_quat'],float); shift.env.sim.forward(); restore(shift,snap); shift._update_observables(force=True); obs_s=shift.env._get_observations()
                comp=compare(snap,snapshot(shift),tol={'body_xpos':1e-3,'eef_pos':1e-3,'body_xquat_rad':2e-3}); cam=camera_compare(snap,snapshot(shift),expect_equal=True,tol=1e-3)
                pc=policy_observation(obs,language,policy); ps=policy_observation(obs_s,language,policy)
                row={'clean_agentview':pc['images']['agentview'][0].numpy(),'shift_agentview':ps['images']['agentview'][0].numpy(),
                     'clean_eye':pc['images']['eye_in_hand'][0].numpy(),'shift_eye':ps['images']['eye_in_hand'][0].numpy(),
                     'state8':state_from_observation(obs,policy.normalizer),'z0':np.random.RandomState(args.z0_seed+100003*sid).randn(1,12,7).astype(np.float32)[0]}
                rows.append(row); meta.append({'role':role,'state_id':int(sid),'pairing_pass':bool(comp['all_pass'] and cam['pass'])})
                print(json.dumps(meta[-1]),flush=True)
            finally: clean.close(); shift.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.out,**{k:np.stack([r[k] for r in rows]) for k in rows[0]}); Path(str(args.out)+'.manifest.json').write_text(json.dumps({'schema':'rgb_oracle_pairs_v2','language':language,'records':meta},indent=2)+'\n')

if __name__=='__main__': main()
