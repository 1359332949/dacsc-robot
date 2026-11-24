"""P1: Resampler attention vs projected bowl/finger masks (eye-in-hand)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/hy-tmp/patch-diag')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'repos/CoRP')); sys.path.insert(0,str(ROOT/'repos/CoRP/scripts'))
from corp import load_policy  # noqa
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation  # noqa
from target_only_action_validation import configure  # noqa
from dacsc.state_pairing import snapshot, restore, compare, camera_compare  # noqa
from dacsc.tta_protocol import load_split, validate_split_identity  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa
from scripts.gate1a2_attention_diagnostic import resampler_tokens_attention  # noqa


def project(sim, camid, p, W=256,H=256):
    cp=np.asarray(sim.data.cam_xpos[camid],float); R=np.asarray(sim.data.cam_xmat[camid],float).reshape(3,3)
    pc=R.T@(p-cp)
    if pc[2]>=-1e-6: return None
    f=H/(2*np.tan(np.deg2rad(sim.model.cam_fovy[camid])/2))
    u=W/2 + f*pc[0]/(-pc[2]); v=H/2 - f*pc[1]/(-pc[2])
    return (u,v)


def masks(env,target_boxes,pads):
    camid=env.sim.model.camera_name2id('robot0_eye_in_hand')
    tm=np.zeros((16,16),bool); fm=np.zeros((16,16),bool)
    # target boxes as projected bbox union
    us=[]; vs=[]
    for bi in target_boxes:
        c=np.asarray(env.sim.data.geom_xpos[bi],float); R=np.asarray(env.sim.data.geom_xmat[bi],float).reshape(3,3); s=np.asarray(env.sim.model.geom_size[bi],float)
        for sx in (-1,1):
            for sy in (-1,1):
                for sz in (-1,1):
                    q=project(env.sim,camid,c+R@(s*np.array([sx,sy,sz])))
                    if q: us.append(q[0]); vs.append(q[1])
    if us: tm[max(0,int(min(vs)//16)):min(16,int(max(vs)//16)+1),max(0,int(min(us)//16)):min(16,int(max(us)//16)+1)]=True
    for pi in pads:
        q=project(env.sim,camid,np.asarray(env.sim.data.geom_xpos[pi],float))
        if q:
            u,v=int(q[0]//16),int(q[1]//16)
            fm[max(0,v-1):min(16,v+2),max(0,u-1):min(16,u+2)]=True
    return tm,fm


def mass(attn,mask):
    if attn.dim()==4: w=attn.mean(dim=(0,1,2))
    else: w=attn.mean(dim=(0,1))
    w=w.reshape(16,16).detach().cpu().numpy(); w=w/(w.sum()+1e-12)
    return float(w[mask].sum()), w


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--state-ids',default='43,45'); ap.add_argument('--steps',type=int,default=20)
    ap.add_argument('--env-seed',type=int,default=7777); ap.add_argument('--z0-seed',type=int,default=7777)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    split=load_split(args.split_json); validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id)
    variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    device=torch.device(args.device); policy=load_policy(args.checkpoint,device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    rs=policy.representation_extractor.cross_attention_resampler; eye_idx=rs.view_names.index('eye_in_hand')
    out={'schema':'attention_mask_audit_v1','records':[]}
    for sid in [int(x) for x in args.state_ids.split(',') if x.strip()]:
        clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        try:
            clean.seed(args.env_seed); clean.reset(); obs=clean.set_init_state(states[sid]); shift.seed(args.env_seed); shift.reset()
            for _ in range(SETTLING_STEPS): obs,_,_,_=clean.step(DUMMY_ACTION)
            body_names=[str(x) for x in clean.sim.model.body_names]; geom_names=[str(x) for x in clean.sim.model.geom_names]
            tid=body_names.index('akita_black_bowl_1_main')
            boxes=[i for i in range(len(geom_names)) if int(clean.sim.model.geom_bodyid[i])==tid and int(clean.sim.model.geom_type[i])==6]
            pads=[i for i,n in enumerate(geom_names) if 'finger' in n and 'pad_collision' in n]
            for step in range(args.steps):
                po=policy_observation(obs,language,policy)
                torch.manual_seed(args.z0_seed+100003*sid+step)
                z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
                clean.sim.forward(); clean._update_observables(force=True); snap=snapshot(clean)
                shift.seed(args.env_seed); shift.reset(); shift.env.sim.model.body_pos[:]=np.asarray(snap['model_signature']['body_pos'],float); shift.env.sim.model.body_quat[:]=np.asarray(snap['model_signature']['body_quat'],float); shift.env.sim.forward(); restore(shift,snap); shift._update_observables(force=True); obs_s=shift.env._get_observations()
                enc_c=encode_observation(policy,po,z0); enc_s=encode_observation(policy,policy_observation(obs_s,language,policy),z0)
                _,attn_c,_,_=resampler_tokens_attention(rs,{v:enc_c['patches'][v] for v in enc_c['patches']},enc_c['pooled_language'])
                _,attn_s,_,_=resampler_tokens_attention(rs,{v:enc_s['patches'][v] for v in enc_s['patches']},enc_s['pooled_language'])
                tm,fm=masks(clean,boxes,pads)
                cm,_=mass(attn_c[eye_idx],tm); cmf,_=mass(attn_c[eye_idx],fm)
                sm,_=mass(attn_s[eye_idx],tm); smf,_=mass(attn_s[eye_idx],fm)
                wc=attn_c[eye_idx].mean(dim=tuple(range(attn_c[eye_idx].dim()-1))).reshape(-1).detach().cpu().numpy(); ws=attn_s[eye_idx].mean(dim=tuple(range(attn_s[eye_idx].dim()-1))).reshape(-1).detach().cpu().numpy()
                topc=set(np.argsort(wc)[-64:]); tops=set(np.argsort(ws)[-64:])
                out['records'].append({'state_id':sid,'step':step,'clean_target_mass':cm,'shift_target_mass':sm,'clean_finger_mass':cmf,'shift_finger_mass':smf,
                                       'clean_bg_mass':1-cm-cmf,'shift_bg_mass':1-sm-smf,'top64_jaccard':len(topc&tops)/len(topc|tops),'target_mask_cells':int(tm.sum()),'finger_mask_cells':int(fm.sum())})
                act=enc_c['controller_actions'][0,0].detach().cpu().numpy()
                obs,reward,done,info=clean.step(np.clip(act,-1,1))
                if done: break
            print(json.dumps({'state':sid,'records':len(out['records'])}) )
        finally: clean.close(); shift.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__': main()
