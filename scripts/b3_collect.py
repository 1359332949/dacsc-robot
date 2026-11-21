"""B3 unified boundary collector: per-view tokens, clipped actions, executed steps."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation  # noqa
from corp import load_policy  # noqa
from target_only_action_validation import configure,sample_z,visual_from_obs  # noqa

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--states',default='34,43,45'); ap.add_argument('--seed',type=int,default=7777); ap.add_argument('--chunks',type=int,default=3); ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292); ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config')); ap.add_argument('--render-gpu',type=int,default=0); args=ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite); variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id); states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); lang=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']; policy=load_policy(args.checkpoint,device=torch.device('cuda:0')); policy.eval(); rows=[]
    for sid in [int(x) for x in args.states.split(',')]:
        for dom,bddl in (('clean',variant['base_bddl']),('shift',variant['env_bddl_name'])):
            env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
            try:
                np.random.seed(args.seed); torch.manual_seed(args.seed); env.seed(args.seed); env.reset(); obs=env.set_init_state(states[sid])
                for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
                g=torch.Generator(device='cuda:0'); g.manual_seed(args.seed); zs=[torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=g) for _ in range(64)]
                for c in range(args.chunks):
                    po=policy_observation(obs,lang,policy); vis,rep=visual_from_obs(policy,po); tok=vis[0].detach().cpu().numpy(); s0=np.asarray(po['state'][0]); acts=policy.normalizer.unnormalize_action(sample_z(policy,rep,zs[c]))[0].float().cpu().numpy(); clipped=np.clip(acts,-1,1); executed=0; success=False
                    for k in range(12):
                        obs,rew,done,info=env.step(clipped[k]); executed+=1; success=bool(is_success(env,info,rew,done)); terminated=bool(success or done)
                        if terminated: break
                    po2=policy_observation(obs,lang,policy); vis2,_=visual_from_obs(policy,po2); tok2=vis2[0].detach().cpu().numpy(); s1=np.asarray(po2['state'][0])
                    rows.append({'state_id':sid,'domain':dom,'chunk':c,'state8_start':s0.tolist(),'state8_end':s1.tolist(),'dstate8':(s1-s0).tolist(),'actions_planned':acts.tolist(),'actions_executed':clipped[:executed].tolist(),'executed_steps':executed,'z0_hash':__import__('hashlib').sha256(zs[c].cpu().numpy().tobytes()).hexdigest(),'view_agent_start':tok[:48].tolist(),'view_eye_start':tok[48:].tolist(),'view_agent_end':tok2[:48].tolist(),'view_eye_end':tok2[48:].tolist(),'terminated':bool(terminated),'success':success})
                    if terminated: break
            finally: env.close()
            print(json.dumps({'state':sid,'domain':dom,'rows':sum(1 for r in rows if r['state_id']==sid and r['domain']==dom)}),flush=True)
    args.out.write_text(json.dumps({'schema':'b3_boundary_v1','rows':rows},indent=2)+'\n')
if __name__=='__main__': main()
