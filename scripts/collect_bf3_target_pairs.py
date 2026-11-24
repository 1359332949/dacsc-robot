"""B-Flow-3: target-dev first-decision RGB pairs with P0 z0 schedules."""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np, torch

ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts'),str(ROOT)]
from corp import load_policy  # noqa
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation, state_from_observation  # noqa
from target_only_action_validation import configure  # noqa
from dacsc.state_pairing import snapshot, restore, compare, camera_compare  # noqa
from dacsc.tta_protocol import load_split, validate_split_identity  # noqa


def make_z0_schedule(seed,count=64,horizon=12,dim=7,device='cuda:0'):
    gen=torch.Generator(device=device); gen.manual_seed(seed)
    return [torch.randn((1,horizon,dim),device=device,dtype=torch.float32,generator=gen) for _ in range(count)]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--ledger',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    split=load_split(args.split_json); matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id); variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    ledger=json.loads(args.ledger.read_text()); policy=load_policy(args.checkpoint,device=torch.device(args.device)); policy.eval()
    rows=[]; meta=[]
    for rec in ledger['records']:
        sid=int(rec['state_id']); seed=int(rec['seed']); z=make_z0_schedule(seed,device=args.device); h=hashlib.sha256(b''.join(x.cpu().numpy().tobytes() for x in z)).hexdigest()
        # verify against baseline clean/shift hashes
        ok=bool(h==rec['clean_z0_schedule_sha256']==rec['shift_z0_schedule_sha256'])
        if not ok: raise RuntimeError(f'z0 hash mismatch {sid}/{seed}: {h}')
        clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu); shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        try:
            np.random.seed(seed); torch.manual_seed(seed); clean.seed(seed); clean.reset(); obs=clean.set_init_state(states[sid]); shift.seed(seed); shift.reset()
            for _ in range(SETTLING_STEPS): obs,_,_,_=clean.step(DUMMY_ACTION)
            clean.sim.forward(); clean._update_observables(force=True); obs=clean.env._get_observations(); snap=snapshot(clean)
            shift.seed(seed); shift.reset(); shift.env.sim.model.body_pos[:]=np.asarray(snap['model_signature']['body_pos'],float); shift.env.sim.model.body_quat[:]=np.asarray(snap['model_signature']['body_quat'],float); shift.env.sim.forward(); restore(shift,snap); shift._update_observables(force=True); obs_s=shift.env._get_observations()
            comp=compare(snap,snapshot(shift),tol={'body_xpos':1e-3,'eef_pos':1e-3,'body_xquat_rad':2e-3}); cam=camera_compare(snap,snapshot(shift),expect_equal=True,tol=1e-3)
            pc=policy_observation(obs,language,policy); ps=policy_observation(obs_s,language,policy)
            sc=state_from_observation(obs,policy.normalizer); ss=state_from_observation(obs_s,policy.normalizer)
            rows.append({'clean_agentview':pc['images']['agentview'][0].numpy(),'shift_agentview':ps['images']['agentview'][0].numpy(),'clean_eye':pc['images']['eye_in_hand'][0].numpy(),'shift_eye':ps['images']['eye_in_hand'][0].numpy(),'state8':sc,'z0':z[0][0].detach().cpu().numpy()})
            meta.append({'state_id':sid,'seed':seed,'clean_success':bool(rec['clean_success']),'shift_success':bool(rec['shift_success']),'z0_schedule_sha256':h,'pairing_pass':bool(comp['all_pass'] and cam['pass']),'state8_max_abs_diff':float(np.max(np.abs(sc-ss)))})
            print(json.dumps(meta[-1]),flush=True)
        finally: clean.close(); shift.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.out,**{k:np.stack([r[k] for r in rows]) for k in rows[0]}); Path(str(args.out)+'.manifest.json').write_text(json.dumps({'schema':'bf3_target_pairs_v1','language':language,'records':meta},indent=2)+'\n')

if __name__=='__main__': main()
