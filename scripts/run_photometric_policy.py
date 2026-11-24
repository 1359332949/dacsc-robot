"""P3: fixed photometric input transform paired clean/shift pilot."""
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
from dacsc.differentiable_policy import _encode_frozen, sample_from_z0  # noqa
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa


def fit_params(path, view):
    d=json.loads(Path(path).read_text()); rs=[r for r in d['records'] if r['role']=='source_fit' and r['view']==view and r.get('pairing_pass')]
    by={}
    for r in rs: by.setdefault(int(r['state_id']),[]).append(r)
    if not rs: raise RuntimeError(f'no source-fit photometric records for {view}')
    cm=[]; cs=[]; sm=[]; ss=[]
    for sid,rows in by.items():
        cm.append(np.mean([r['clean']['lum_mean'] for r in rows])); cs.append(np.mean([r['clean']['lum_std'] for r in rows]))
        sm.append(np.mean([r['shift']['lum_mean'] for r in rows])); ss.append(np.mean([r['shift']['lum_std'] for r in rows]))
    cm,cs,sm,ss=map(np.mean,(cm,cs,sm,ss))
    if not np.isfinite([cm,cs,sm,ss]).all(): raise RuntimeError(f'non-finite fit params for {view}')
    a=float(np.clip(cs/(ss+1e-6),0.7,1.6)); b=float(np.clip(cm-a*sm,-0.3,0.3))
    return {'a':a,'b':b,'clean_lum_mean':float(cm),'shift_lum_mean':float(sm),'clean_lum_std':float(cs),'shift_lum_std':float(ss)}


def apply_transform(obs, params):
    out=dict(obs); clip_stats={}
    for view,key in (('agentview','agentview_image'),('eye_in_hand','robot0_eye_in_hand_image')):
        p=params[view]; img=np.asarray(obs[key],dtype=np.float32)/255.0
        raw=p['a']*img+p['b']; img=np.clip(raw,0,1)
        clip_stats[view]=float(((raw<0)|(raw>1)).mean())
        out[key]=(img*255.0).round().astype(np.uint8)
    return out, clip_stats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--photometric-fit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--domains',default='clean,shift')
    ap.add_argument('--state-ids',default='36,37,43,45'); ap.add_argument('--env-seed',type=int,default=7777); ap.add_argument('--z0-seed',type=int,default=7777)
    ap.add_argument('--modes',default='frozen,transform'); ap.add_argument('--executed-k',type=int,default=12); ap.add_argument('--max-steps',type=int,default=220)
    ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292)
    ap.add_argument('--device',default='cuda:0'); ap.add_argument('--render-gpu',type=int,default=0)
    ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config'))
    args=ap.parse_args()
    domains=[x.strip() for x in args.domains.split(',') if x.strip()]
    modes=[x.strip() for x in args.modes.split(',') if x.strip()]
    if not domains or not set(domains) <= {'clean','shift'}: raise ValueError(f'invalid domains {domains}')
    if not modes or not set(modes) <= {'frozen','transform'}: raise ValueError(f'invalid modes {modes}')
    if args.executed_k <= 0 or args.max_steps <= 0: raise ValueError('executed_k/max_steps must be positive')
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite)
    split=load_split(args.split_json); validate_split_identity(split,args.suite,task.get('base_task') or '',args.lighting_id)
    variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id)
    states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']
    ids=resolve_indices(split,'target_dev',args.state_ids); device=torch.device(args.device); policy=load_policy(args.checkpoint,device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    params={v:fit_params(args.photometric_fit,v) for v in ('agentview','eye_in_hand')}
    records=[]
    for domain in domains:
        bddl=variant['env_bddl_name'] if domain=='shift' else variant['base_bddl']
        for mode in modes:
            for state_id in ids:
                env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
                try:
                    np.random.seed(args.env_seed); torch.manual_seed(args.env_seed); env.seed(args.env_seed); env.reset(); obs=env.set_init_state(states[state_id])
                    for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
                    steps=0; chunk=0; success=False; done=False
                    transform_clip=[]; action_clip=[]
                    while not success and not done and steps<args.max_steps:
                        if mode=='frozen':
                            use=obs
                        else:
                            use,clip_stats=apply_transform(obs,params); transform_clip.append(clip_stats)
                        po=policy_observation(use,language,policy)
                        torch.manual_seed(args.z0_seed+100003*int(state_id)+chunk)
                        z0=torch.randn((1,policy.action_generator.action_horizon,policy.action_generator.action_dim),device=device,dtype=next(policy.parameters()).dtype)
                        pieces=_encode_frozen(policy,po)
                        rep=Representation(tokens=torch.cat([pieces['visual_tokens'],pieces['language_tokens'].to(pieces['visual_tokens'].dtype),pieces['state_token']],dim=1),pooled_language=pieces['pooled_language'],action_dtype=pieces['visual_tokens'].dtype)
                        with torch.no_grad():
                            ctrl=policy.normalizer.unnormalize_action(sample_from_z0(policy,rep,z0))
                        if ctrl.shape[1] < args.executed_k:
                            raise RuntimeError(f'action horizon {ctrl.shape[1]} < executed_k {args.executed_k}')
                        action_clip.append(float((ctrl.abs()>1).float().mean()))
                        for k in range(min(args.executed_k,ctrl.shape[1])):
                            obs,reward,done,info=env.step(np.clip(ctrl[0,k].detach().cpu().numpy(),-1,1)); steps+=1
                            success=is_success(env,info,reward,done)
                            if success or done or steps>=args.max_steps: break
                        chunk+=1
                    records.append({'state_id':int(state_id),'domain':domain,'mode':mode,'success':bool(success),'steps':int(steps),
                                    'hit_max_steps':bool(steps>=args.max_steps and not success),'transform':params,
                                    'transform_clip_fraction_mean':{v:float(np.mean([c[v] for c in transform_clip])) if transform_clip else None for v in ('agentview','eye_in_hand')},
                                    'action_clip_fraction_mean':float(np.mean(action_clip)) if action_clip else None})
                    print(json.dumps(records[-1]),flush=True)
                finally: env.close()
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({'schema':'photometric_policy_pilot_v1','params':params,'records':records},indent=2)+'\n')
    print(json.dumps({'out':str(args.out),'n':len(records)}))

if __name__=='__main__': main()
