"""Build shared source-domain calibration from 10 libero_spatial Clean init0 trajectories."""
from __future__ import annotations
import argparse,json,sys,os,ctypes,random
from pathlib import Path
import numpy as np,torch
ROOT=Path('/hy-tmp/patch-diag'); CORP=ROOT/'repos/CoRP'; LIB=Path('/hy-tmp/LIBERO-plus')
sys.path[:0]=[str(ROOT),str(CORP),str(CORP/'scripts'),str(LIB)]
os.environ['MUJOCO_GL']='egl'; os.environ['PYOPENGL_PLATFORM']='egl'
from corp import load_policy
from corp.preprocessing import prepare_observation
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation
from target_only_action_validation import sample_z,visual_from_obs
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import grab_language_from_filename
from dacsc.static_visual_v2 import seed_for
OUT=ROOT/'runs/phase2_calib'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task-index',type=int,required=True); ap.add_argument('--max-steps',type=int,default=300); a=ap.parse_args()
    man=json.load(open(ROOT/'docs/data/static_visual_v2_suite/final/dev_manifest.json')); cleans=[c for c in man['cases'] if c['factor']=='clean']; c=cleans[a.task_index]; task=c['base_task']
    bddl=LIB/c['bddl']; init=torch.load(LIB/c['init_state_file'],weights_only=False,map_location='cpu')[c['init_state_index']]; lang=grab_language_from_filename('libero_spatial',task+'.bddl')
    policy=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False); ext=policy.representation_extractor
    caps={i:[] for i in range(12)}; hooks=[]
    for i,b in enumerate(ext.visual_encoder.backbone.blocks):
        def mk(i):
            def h(m,a,x): caps[i].append(x[:,1:].detach()); return None
            return h
        hooks.append(b.register_forward_hook(mk(i)))
    pix_sum=torch.zeros(2,3); pix_sq=torch.zeros(2,3); pix_n=0; b3_m=[]; b3_q=[]; b12_m=[]; b12_q=[]; all_m=[torch.zeros(2,384) for _ in range(12)]; all_q=[torch.zeros(2,384) for _ in range(12)]; nsteps=0
    es=seed_for('phase2_calib',task,'env'); random.seed(es);np.random.seed(es);torch.manual_seed(es);ctypes.CDLL(None).srand(ctypes.c_uint(es)); env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
    try:
        env.noise=0;env.seed(es);env.reset();obs=env.set_init_state(init)
        for _ in range(SETTLING_STEPS): obs=env.step(DUMMY_ACTION)[0]
        with torch.no_grad():
            for step in range(a.max_steps):
                po=policy_observation(obs,lang,policy)
                for vi,v in enumerate(ext.view_names):
                    x=po['images'][v].float(); pix_sum[vi]+=x.mean((0,1,2)); pix_sq[vi]+=x.square().mean((0,1,2)); pix_n+=1
                batch=prepare_observation(po,config=policy.config,normalizer=policy.normalizer,device='cuda:0')
                for i in range(12): caps[i].clear()
                ext.visual_encoder(torch.cat([batch['images'][v] for v in ext.view_names]))
                for i in range(12):
                    p=caps[i][0]; m=p.mean(1); q=p.square().mean(1)
                    all_m[i]+=m.cpu(); all_q[i]+=q.cpu()
                    if i==2: b3_m.append(m.cpu()); b3_q.append(q.cpu())
                    if i==11: b12_m.append(m.cpu()); b12_q.append(q.cpu())
                nsteps+=1
                gen=torch.Generator(device='cuda:0').manual_seed(seed_for(task,step,'z')); z=torch.randn((1,12,7),device='cuda:0',generator=gen); _,rep=visual_from_obs(policy,po); plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu().numpy()
                for ac in plan:
                    obs,rew,done,info=env.step(ac)
                    if done: break
                if done: break
    finally:
        for h in hooks: h.remove()
        env.close()
    OUT.mkdir(parents=True,exist_ok=True)
    mu_l=torch.stack([all_m[i]/max(nsteps,1) for i in range(12)]).numpy(); ss_l=torch.sqrt(torch.stack([all_q[i]/max(nsteps,1) for i in range(12)])-torch.from_numpy(mu_l).square()).clamp_min(0).numpy()
    np.savez_compressed(OUT/f'task_{a.task_index}.npz',task=task,language=lang,pixel_sum=pix_sum.numpy(),pixel_sq=pix_sq.numpy(),pixel_n=np.array(pix_n),mu_l=mu_l,ss_l=ss_l,b3_m=torch.stack(b3_m).numpy(),b3_q=torch.stack(b3_q).numpy(),b12_m=torch.stack(b12_m).numpy(),b12_q=torch.stack(b12_q).numpy())
    print(json.dumps({'task':task,'nsteps':nsteps,'pixel_n':pix_n,'b3_frames':len(b3_m)}),flush=True)
if __name__=='__main__': main()
