"""Minimal pre-Resampler prototype: source-anchored patch token normalization."""
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
from dacsc.differentiable_policy import sample_from_z0  # noqa
from dacsc.tta_protocol import load_split, validate_split_identity  # noqa
from scripts.collect_patch_pairs import encode_observation  # noqa


def source_stats(pairs):
    z=np.load(pairs,allow_pickle=False)
    out={}
    for v in ('agentview','eye_in_hand'):
        p=z[f'clean_{v}'].astype(np.float64)
        out[v]={'mean':torch.tensor(p.mean(axis=(0,1)),dtype=torch.float32),'std':torch.tensor(np.maximum(p.std(axis=(0,1)),1e-4),dtype=torch.float32)}
    return out


def adapt_patches(patches, stats, lam=1.0):
    out={}
    for v,p in patches.items():
        a=p.float(); mu=a.mean(dim=1,keepdim=True); sd=a.std(dim=1,keepdim=True).clamp_min(1e-4)
        tgt=stats[v]['mean'].to(a.device).view(1,1,-1); tsd=stats[v]['std'].to(a.device).view(1,1,-1)
        norm=(a-mu)/sd*tsd+tgt
        out[v]=(a+lam*(norm-a)).to(dtype=p.dtype)
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--state-ids',default='34,35,36,37,39,43,44,45,48,49'); ap.add_argument('--domains',default='clean,shift'); ap.add_argument('--lam',type=float,default=1.0)
    ap.add_argument('--env-seed',type=int,default=7777); ap.add_argument('--z0-seed',type=int,default=7777)
    ap.add_argument('--executed-k',type=int,default=12); ap.add_argument('--max-steps',type=int,default=220)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args(); domains=[x.strip() for x in args.domains.split(',') if x.strip()]
    if not domains or not set(domains)<={'clean','shift'} or args.lam<0: raise ValueError('bad args')
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    split=load_split(args.split_json); validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id)
    variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    device=torch.device(args.device); policy=load_policy(args.checkpoint,device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    rs=policy.representation_extractor.cross_attention_resampler; stats=source_stats(args.pairs)
    records=[]
    for domain in domains:
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
                    enc=encode_observation(policy,po,z0); patches=adapt_patches(enc['patches'],stats,args.lam)
                    visual=rs(patches,enc['pooled_language'])
                    rep=Representation(tokens=torch.cat([visual,enc['language_tokens'].to(visual.dtype),enc['state_token']],dim=1),pooled_language=enc['pooled_language'],action_dtype=visual.dtype)
                    with torch.no_grad(): ctrl=policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))
                    for k in range(min(args.executed_k,ctrl.shape[1])):
                        obs,reward,done,info=env.step(np.clip(ctrl[0,k].detach().cpu().numpy(),-1,1)); step+=1
                        success=is_success(env,info,reward,done)
                        if success or done or step>=args.max_steps: break
                    chunk+=1
                records.append({'state_id':sid,'domain':domain,'success':bool(success),'steps':int(step),'lambda':args.lam,'source_stats':'clean source patches'})
                print(json.dumps(records[-1]),flush=True)
            finally: env.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'patch_normalization_v1','records':records},indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'n':len(records)}))

if __name__=='__main__': main()
