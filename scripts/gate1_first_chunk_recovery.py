"""Gate 1: first-chunk recoverability for P0 Shift episodes (E=12)."""
from __future__ import annotations
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,MAX_STEPS,is_success,policy_observation  # noqa
from corp import load_policy  # noqa
from target_only_action_validation import configure,sample_z,visual_from_obs  # noqa
from dacsc.state_pairing import snapshot,restore  # noqa
from dacsc.observation_render import render_policy_observation  # noqa
from dacsc.tta_protocol import load_split  # noqa

def make_z0(seed,count=64):
    g=torch.Generator(device='cuda:0'); g.manual_seed(seed); return [torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=g) for _ in range(count)]
def rollout(env,policy,obs,zs,chunk,steps,language,max_steps=220):
    while steps<max_steps:
        po=policy_observation(obs,language,policy); vis,rep=visual_from_obs(policy,po); z0=zs[min(chunk,63)]; ctrl=policy.normalizer.unnormalize_action(sample_z(policy,rep,z0))[0].float().cpu().numpy()
        for k in range(12):
            obs,reward,done,info=env.step(np.clip(ctrl[k],-1,1)); steps+=1
            if is_success(env,info,reward,done) or done or steps>=max_steps: return bool(is_success(env,info,reward,done)),steps,obs
        chunk+=1
    return False,steps,obs
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--ledger',type=Path,required=True); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--split-json',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--max-steps',type=int,default=220); ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292); ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config')); ap.add_argument('--render-gpu',type=int,default=0); args=ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite); variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id); states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); language=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']; ledger=json.loads(args.ledger.read_text()); policy=load_policy(args.checkpoint,device=torch.device('cuda:0')); policy.eval(); out=[]
    for rec in ledger['records']:
        sid=int(rec['state_id']); seed=int(rec['seed']); zs=make_z0(seed); h=hashlib.sha256(b''.join(x.cpu().numpy().tobytes() for x in zs)).hexdigest(); assert h==rec['clean_z0_schedule_sha256']==rec['shift_z0_schedule_sha256']
        env=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu); clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        try:
            np.random.seed(seed); torch.manual_seed(seed); env.seed(seed); env.reset(); obs=env.set_init_state(states[sid])
            for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
            po=policy_observation(obs,language,policy); vis,rep=visual_from_obs(policy,po); ctrl=policy.normalizer.unnormalize_action(sample_z(policy,rep,zs[0]))[0].float().cpu().numpy(); steps=0
            for k in range(12):
                obs,reward,done,info=env.step(np.clip(ctrl[k],-1,1)); steps+=1
            snap=snapshot(env)
            sh_succ,sh_steps,_=rollout(env,policy,obs,zs,1,steps,language)
            clean_obs=render_policy_observation(clean,snap); cl_succ,cl_steps,_=rollout(clean,policy,clean_obs,zs,1,steps,language)
            out.append({'state_id':sid,'seed':seed,'clean_success':rec['clean_success'],'shift_success':rec['shift_success'],'first12_steps':12,'shift_recovery':sh_succ,'shift_steps':sh_steps,'clean_continuation':cl_succ,'clean_steps':cl_steps,'z0_schedule_sha256':h}); print(json.dumps(out[-1]),flush=True)
        finally: env.close(); clean.close()
    args.out.write_text(json.dumps({'schema':'gate1_first_chunk_recovery_v1','records':out},indent=2)+'\n')
if __name__=='__main__': main()
