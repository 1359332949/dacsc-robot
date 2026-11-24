"""P2: collect paired Clean/Shift raw-image photometric statistics at source states."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/hy-tmp/patch-diag')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'repos/CoRP')); sys.path.insert(0,str(ROOT/'repos/CoRP/scripts'))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation  # noqa
from corp import load_policy  # noqa
from target_only_action_validation import configure  # noqa
from dacsc.state_pairing import snapshot, restore, compare, camera_compare, model_compatible  # noqa
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa


def img_stats(img):
    a=np.asarray(img,dtype=np.float32)
    if a.ndim!=3 or a.shape[2]<3: raise ValueError(f'bad image shape {a.shape}')
    a=a[...,:3]/255.0
    lum=0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
    def q(x,p): return float(np.quantile(x,p))
    sat=(a.max(-1)-a.min(-1))/(a.max(-1)+1e-6)
    gy,gx=np.gradient(lum); edge=np.sqrt(gx*gx+gy*gy)
    # local 8x8 grid luminance/contrast/edge summaries
    h,w=lum.shape; ph,pw=max(1,h//8),max(1,w//8); lm=[]; ls=[]; le=[]
    for yy in range(0,h,ph):
        for xx in range(0,w,pw):
            blk=lum[yy:yy+ph,xx:xx+pw]; be=edge[yy:yy+ph,xx:xx+pw]
            lm.append(float(blk.mean())); ls.append(float(blk.std())); le.append(float(be.mean()))
    return {
        'lum_mean':float(lum.mean()),'lum_median':float(np.median(lum)),'lum_std':float(lum.std()),
        'lum_p10':q(lum,.1),'lum_p90':q(lum,.9),'lum_p95':q(lum,.95),
        'contrast_p90_p10':q(lum,.9)-q(lum,.1),
        'rgb_mean':[float(a[...,i].mean()) for i in range(3)],
        'rgb_std':[float(a[...,i].std()) for i in range(3)],
        'saturation_mean':float(sat.mean()),
        'shadow_fraction':float((lum<0.1).mean()),'highlight_fraction':float((lum>0.9).mean()),
        'edge_mean':float(edge.mean()),'edge_std':float(edge.std()),
        'local_lum_mean_mean':float(np.mean(lm)),'local_lum_std_mean':float(np.mean(ls)),
        'local_lum_std_max':float(np.max(ls)),'local_edge_mean_mean':float(np.mean(le)),
        'local_edge_std':float(np.std(le)),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--roles',default='source_fit'); ap.add_argument('--state-ids',default='')
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--seed',type=int,default=7777); ap.add_argument('--early-chunks',default='0,1')
    ap.add_argument('--late-chunks',default='6,7,8,9'); ap.add_argument('--executed-k',type=int,default=12)
    ap.add_argument('--max-steps',type=int,default=220); ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(x for x in matrix['base_tasks'] if x['suite']==args.suite)
    split=load_split(args.split_json); validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id)
    variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False)
    language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    device=torch.device(args.device); policy=load_policy(args.checkpoint,device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    early={int(x) for x in args.early_chunks.split(',') if x.strip()}; late={int(x) for x in args.late_chunks.split(',') if x.strip()}; selected=early|late
    roles=[x.strip() for x in args.roles.split(',') if x.strip()]; records=[]
    for role in roles:
        ids=resolve_indices(split,role,args.state_ids)
        for state_id in ids:
            shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            try:
                np.random.seed(args.seed); torch.manual_seed(args.seed)
                shift.seed(args.seed); shift.reset(); obs=shift.set_init_state(states[state_id])
                for _ in range(SETTLING_STEPS): obs,_,_,_=shift.step(DUMMY_ACTION)
                steps=0; chunk=0; done=False
                while not done and steps<args.max_steps and chunk<=max(selected):
                    po=policy_observation(obs,language,policy)
                    torch.manual_seed(args.seed+100003*int(state_id)+chunk)
                    z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
                    enc=encode_observation(policy,po,z0); ctrl=enc['controller_actions']
                    if chunk in selected:
                        shift.sim.forward(); shift._update_observables(force=True)
                        obs_s=shift.env._get_observations(); snap=snapshot(shift)
                        clean.seed(args.seed+991*chunk); clean.reset()
                        clean.env.sim.model.body_pos[:]=np.asarray(snap['model_signature']['body_pos'],float)
                        clean.env.sim.model.body_quat[:]=np.asarray(snap['model_signature']['body_quat'],float)
                        clean.env.sim.forward(); restore(clean,snap); clean._update_observables(force=True)
                        obs_c=clean.env._get_observations(); snap_c=snapshot(clean)
                        comp=model_compatible(snap_c,snap,strict_bodies=True,strict_geometry=True,strict_physics=True)
                        compat=compare(snap,snap_c,tol={'body_xpos':1e-3,'eef_pos':1e-3,'body_xquat_rad':2e-3})
                        cam=camera_compare(snap,snap_c,expect_equal=True,tol=1e-3)
                        window='early' if chunk in early else 'late'
                        for view,key in (('agentview','agentview_image'),('eye_in_hand','robot0_eye_in_hand_image')):
                            sc=img_stats(obs_c[key]); ss=img_stats(obs_s[key])
                            records.append({'role':role,'state_id':int(state_id),'chunk':int(chunk),'step':int(steps),'window':window,
                                            'view':view,'clean':sc,'shift':ss,
                                            'pairing_pass':bool(comp['pass'] and compat['all_pass'] and cam['pass']),
                                            'model_compat':bool(comp['pass']),'state_compare':bool(compat['all_pass']),'camera_compare':bool(cam['pass'])})
                    for k in range(min(args.executed_k,ctrl.shape[1])):
                        obs,reward,done,info=shift.step(np.clip(ctrl[0,k].detach().cpu().numpy(),-1,1)); steps+=1
                        if done or steps>=args.max_steps: break
                    chunk+=1
                print(json.dumps({'role':role,'state_id':int(state_id),'chunks':chunk,'steps':steps,'records':len(records)}),flush=True)
            finally:
                shift.close(); clean.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'photometric_pairs_v1','roles':roles,'records':records},indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'n':len(records)}))

if __name__=='__main__': main()
