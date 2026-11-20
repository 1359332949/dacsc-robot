"""Prototype 2: local patch-energy calibration with hard residual cap."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from corp import load_policy  # noqa
from corp.representation.extractor import Representation  # noqa
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa
from target_only_action_validation import configure  # noqa
from dacsc.differentiable_policy import sample_from_z0  # noqa
from dacsc.tta_protocol import load_split, validate_split_identity  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa


def source_norm(pairs,view):
    z=np.load(pairs,allow_pickle=False); p=z[f'clean_{view}'].astype(np.float32)
    n=np.linalg.norm(p,axis=-1)  # [N,256]
    return float(n.mean()), float(n.std())


def adapt(patches, means, eps=0.2):
    out={}
    for v,p in patches.items():
        a=p.float(); n=a.norm(dim=-1,keepdim=True).clamp_min(1e-6); target=means[v]
        factor=1.0+eps*torch.tanh(torch.log(torch.tensor(target,device=a.device)/(n+1e-6)))
        out[v]=(a*factor).to(p.dtype)
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--state-ids',default='34,35,36,37,43,44,45,49'); ap.add_argument('--domains',default='clean')
    ap.add_argument('--eps',type=float,default=0.2); ap.add_argument('--env-seed',type=int,default=7777); ap.add_argument('--z0-seed',type=int,default=7777)
    ap.add_argument('--executed-k',type=int,default=12); ap.add_argument('--max-steps',type=int,default=220)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args(); configure(args)
    if not (0<args.eps<=0.5): raise ValueError('eps must be in (0,0.5]')
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    split=load_split(args.split_json); validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id)
    variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    device=torch.device(args.device); policy=load_policy(args.checkpoint,device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    rs=policy.representation_extractor.cross_attention_resampler
    means={v:source_norm(args.pairs,v)[0] for v in ('agentview','eye_in_hand')}
    records=[]
    for domain in [x.strip() for x in args.domains.split(',') if x.strip()]:
        bddl=variant['env_bddl_name'] if domain=='shift' else variant['base_bddl']
        for sid in [int(x) for x in args.state_ids.split(',') if x.strip()]:
            env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            try:
                np.random.seed(args.env_seed); torch.manual_seed(args.env_seed); env.seed(args.env_seed); env.reset(); obs=env.set_init_state(states[sid])
                for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
                step=0; chunk=0; success=False; done=False
                while not success and not done and step<args.max_steps:
                    po=policy_observation(obs,language,policy)
                    torch.manual_seed(args.z0_seed+100003*sid+chunk)
                    z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
                    enc=encode_observation(policy,po,z0); patches=adapt(enc['patches'],means,args.eps)
                    visual=rs(patches,enc['pooled_language'])
                    rep=Representation(tokens=torch.cat([visual,enc['language_tokens'].to(visual.dtype),enc['state_token']],dim=1),pooled_language=enc['pooled_language'],action_dtype=visual.dtype)
                    with torch.no_grad(): ctrl=policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))
                    for k in range(min(args.executed_k,ctrl.shape[1])):
                        obs,reward,done,info=env.step(np.clip(ctrl[0,k].detach().cpu().numpy(),-1,1)); step+=1
                        success=is_success(env,info,reward,done)
                        if success or done or step>=args.max_steps: break
                    chunk+=1
                records.append({'state_id':sid,'domain':domain,'success':bool(success),'steps':int(step),'eps':args.eps,'mode':'local_patch_energy'})
                print(json.dumps(records[-1]),flush=True)
            finally: env.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'local_patch_energy_v1','records':records},indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'n':len(records)}))

if __name__=='__main__': main()
