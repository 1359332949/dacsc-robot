"""B2: same physical state + same clipped actions in Clean/Shift; compare physics and visual tokens."""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation  # noqa
from corp import load_policy  # noqa
from target_only_action_validation import configure,sample_z,visual_from_obs  # noqa
from dacsc.state_pairing import snapshot,restore,model_compatible,compare  # noqa

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--states',default='34,43,45'); ap.add_argument('--seed',type=int,default=7777); ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292); ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config')); ap.add_argument('--render-gpu',type=int,default=0); args=ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite); variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id); states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); lang=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']; policy=load_policy(args.checkpoint,device=torch.device('cuda:0')); policy.eval(); out=[]
    for sid in [int(x) for x in args.states.split(',')]:
        clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu); shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        try:
            np.random.seed(args.seed); torch.manual_seed(args.seed); clean.seed(args.seed); clean.reset(); obs_c=clean.set_init_state(states[sid]); shift.seed(args.seed); shift.reset(); obs_s=shift.set_init_state(states[sid])
            for _ in range(SETTLING_STEPS): obs_c,_,_,_=clean.step(DUMMY_ACTION); obs_s,_,_,_=shift.step(DUMMY_ACTION)
            po=policy_observation(obs_s,lang,policy); vis,rep=visual_from_obs(policy,po); g=torch.Generator(device='cuda:0'); g.manual_seed(args.seed); z0=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=g); acts=policy.normalizer.unnormalize_action(sample_z(policy,rep,z0))[0].float().cpu().numpy(); clipped=np.clip(acts,-1,1); act_hash=hashlib.sha256(clipped.tobytes()).hexdigest(); pc=[]; ps=[]
            phys=[]
            for k in range(12):
                obs_c,_,_,_=clean.step(clipped[k]); obs_s,_,_,_=shift.step(clipped[k]); pco=policy_observation(obs_c,lang,policy); pso=policy_observation(obs_s,lang,policy); pc.append(pco['state'][0]); ps.append(pso['state'][0]); sc=snapshot(clean); ss=snapshot(shift); d={}
                for key in ('qpos','qvel','act','body_xpos','body_xquat','eef_pos'):
                    if key in sc and key in ss and np.asarray(sc[key]).size>0 and np.asarray(ss[key]).size>0: d[key]=float(np.max(np.abs(np.asarray(sc[key])-np.asarray(ss[key]))))
                phys.append(d)
            po_c=policy_observation(obs_c,lang,policy); po_s=policy_observation(obs_s,lang,policy); vc,rc=visual_from_obs(policy,po_c); vs,rs=visual_from_obs(policy,po_s); tc=vc[0].detach().cpu().numpy(); ts=vs[0].detach().cpu().numpy(); nc=tc.shape[0]//2; view_cos=[]; view_rms=[]
            for v in range(2):
                a=tc[v*48:(v+1)*48]; b=ts[v*48:(v+1)*48]; view_cos.append(float(np.mean(np.sum(a*b,axis=-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+1e-9)))); view_rms.append(float(np.sqrt(np.mean((a-b)**2))))
            pcmax=float(np.max(np.abs(np.asarray(pc)-np.asarray(ps)))); pmax=max((v for d in phys for v in d.values()),default=None); compat=model_compatible(snapshot(clean),snapshot(shift),strict_bodies=False,strict_geometry=False,strict_physics=False); out.append({'state_id':sid,'action_hash':act_hash,'policy_state8_maxdiff':pcmax,'full_physics_maxdiff':pmax,'physics_per_key':{k:(max(d.get(k,0.0) for d in phys) if any(k in d for d in phys) else None) for k in ('qpos','qvel','act','body_xpos','body_xquat','eef_pos')},'state_mapping_compatible':bool(compat.get('pass',False)) if isinstance(compat,dict) else bool(compat),'scene_model_compatible':bool(model_compatible(snapshot(clean),snapshot(shift),strict_bodies=True).get('pass',False)),'view_cos':view_cos,'view_rms':view_rms,'clean_state8':np.asarray(pc)[-1].tolist(),'shift_state8':np.asarray(ps)[-1].tolist()}); print(json.dumps({'state':sid,'policy_maxdiff':pcmax,'physics_maxdiff':pmax,'state_map':out[-1]['state_mapping_compatible'],'scene_model':out[-1]['scene_model_compatible'],'view_cos':view_cos,'view_rms':view_rms}),flush=True)
        finally: clean.close(); shift.close()
    args.out.write_text(json.dumps({'schema':'b2_replay_control_v1','records':out},indent=2)+'\n')
if __name__=='__main__': main()
