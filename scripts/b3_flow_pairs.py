"""B3-2 pilot: same-x Clean/Shift flow velocity residuals on 3 states."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation  # noqa
from corp import load_policy  # noqa
from target_only_action_validation import configure,visual_from_obs  # noqa

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--matrix',type=Path,required=True); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--states',default='34,43,45'); ap.add_argument('--z0-count',type=int,default=4); ap.add_argument('--steps',type=int,default=10); ap.add_argument('--seed',type=int,default=7777); ap.add_argument('--suite',default='libero_spatial'); ap.add_argument('--lighting-id',type=int,default=2292); ap.add_argument('--libero-src',type=Path,default=Path('/hy-tmp/libero_plus')); ap.add_argument('--libero-config',type=Path,default=Path('/hy-tmp/libero_config')); ap.add_argument('--render-gpu',type=int,default=0); args=ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix=json.loads(args.matrix.read_text()); task=next(b for b in matrix['base_tasks'] if b['suite']==args.suite); variant=next(v for v in task['variants']['lighting'] if int(v['classification_id'])==args.lighting_id); states=torch.load(task['state_split']['source_file'],map_location='cpu',weights_only=False); lang=BDDLUtils.get_problem_info(variant['base_bddl'])['language_instruction']; policy=load_policy(args.checkpoint,device=torch.device('cuda:0')); policy.eval(); rows=[]; dt=1/args.steps
    for sid in [int(x) for x in args.states.split(',')]:
        clean=OffScreenRenderEnv(bddl_file_name=variant['base_bddl'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu); shift=OffScreenRenderEnv(bddl_file_name=variant['env_bddl_name'],camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        try:
            np.random.seed(args.seed); torch.manual_seed(args.seed); clean.seed(args.seed); clean.reset(); oc=clean.set_init_state(states[sid]); shift.seed(args.seed); shift.reset(); os_=shift.set_init_state(states[sid])
            for _ in range(SETTLING_STEPS): oc,_,_,_=clean.step(DUMMY_ACTION); os_,_,_,_=shift.step(DUMMY_ACTION)
            poc=policy_observation(oc,lang,policy); pos=policy_observation(os_,lang,policy); zc,rc=visual_from_obs(policy,poc); zs,rs=visual_from_obs(policy,pos)
            # use policy predict path components
            P=policy.representation_extractor; batchc=__import__('corp.preprocessing',fromlist=['prepare_observation']).prepare_observation(poc,config=policy.config,normalizer=policy.normalizer,device='cuda:0'); batchs=__import__('corp.preprocessing',fromlist=['prepare_observation']).prepare_observation(pos,config=policy.config,normalizer=policy.normalizer,device='cuda:0'); Rc=P(**batchc); Rs=P(**batchs); prc=policy.action_generator.prepare_prefix(Rc); prs=policy.action_generator.prepare_prefix(Rs); ag=policy.action_generator
            def hv(prefix,x,tau):
                t=torch.full((x.shape[0],),tau,device='cuda:0',dtype=x.dtype); a=ag.action_projection(x)+ag.action_position[:,:x.shape[1]]+ag.time_embedding(t).unsqueeze(1)
                for b,c in zip(ag.blocks,prefix): a=b(a,c)
                h=ag.output_norm(a); return h,ag.velocity_projection(h)
            for zi in range(args.z0_count):
                g=torch.Generator(device='cuda:0'); g.manual_seed(500000+zi*1009+sid); x=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=g)
                for n in range(args.steps):
                    tau=n*dt; h_s,v_s=hv(prs,x,tau); h_c,v_c=hv(prc,x,tau); rec=v_c-v_s
                    for k in range(12): rows.append({'state_id':sid,'z0_idx':zi,'flow_step':n,'tau':tau,'k':k,'h':h_s[0,k].detach().cpu().numpy().tolist(),'z_agent':zs[0,:48].mean(0).detach().cpu().numpy().tolist(),'z_eye':zs[0,48:].mean(0).detach().cpu().numpy().tolist(),'z_clean_agent':zc[0,:48].mean(0).detach().cpu().numpy().tolist(),'z_clean_eye':zc[0,48:].mean(0).detach().cpu().numpy().tolist(),'state8':batchs['state'][0].detach().cpu().numpy().tolist(),'r':rec[0,k].detach().cpu().numpy().tolist()})
                    x=x+v_s*dt
            print(json.dumps({'state':sid,'n':sum(1 for r in rows if r['state_id']==sid)}),flush=True)
        finally: clean.close(); shift.close()
    args.out.write_text(json.dumps({'schema':'b3_flow_pairs_v1','rows':rows},indent=2)+'\n')
if __name__=='__main__': main()
