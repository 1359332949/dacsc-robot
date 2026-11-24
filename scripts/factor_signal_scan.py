"""t=0 pixel + intrinsic flow signal scan across all 3 factors and Clean."""
from __future__ import annotations
import argparse,json,sys,os
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, cv2
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),'/hy-tmp/LIBERO-plus',str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
os.environ.setdefault('MUJOCO_GL','osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation  # noqa
from corp import load_policy  # noqa
from target_only_action_validation import sample_z, visual_from_obs  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa
import libero.libero.envs.bddl_utils as BDDLUtils  # noqa
from dacsc.static_visual_v2 import sensor_transform, seed_for, array_hash  # noqa
import libero.libero.envs.env_wrapper as noise_wrapper  # noqa

def state_from_flat(sim,st):
    st=np.asarray(st).reshape(-1); nq=int(sim.model.nq); nv=int(sim.model.nv)
    sim.data.time=float(st[0]); sim.data.qpos[:]=st[1:1+nq]; sim.data.qvel[:]=st[1+nq:1+nq+nv]; sim.forward()

def hf_ratio(gray):
    g=gray.astype(np.float32)/255.; f=np.fft.fftshift(np.fft.fft2(g)); a=np.abs(f); h,w=a.shape; yy,xx=np.mgrid[0:h,0:w]; r=np.sqrt(((yy-h/2)/(h/2))**2+((xx-w/2)/(w/2))**2); return float(a[r>0.5].mean()/(a.mean()+1e-9))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--verified',required=True); ap.add_argument('--results',required=True); ap.add_argument('--out',required=True); ap.add_argument('--m',type=int,default=8); ap.add_argument('--render-gpu',type=int,default=0); args=ap.parse_args()
    man=json.load(open(args.manifest)); ver=json.load(open(args.verified)); succ={r['case_id']:bool(r['success']) for r in ver['results']}; cases=man['cases']; res={}
    for p in Path(args.results).glob('*.json'):
        try:
            d=json.loads(p.read_text(errors='ignore')); res[d['case_id']]=d
        except Exception:
            pass
    policy=load_policy(args.checkpoint,device='cuda:0'); policy.eval(); rows=[]
    for c in cases:
        bddl=str(Path('/hy-tmp/LIBERO-plus')/c['bddl']); init=str(Path('/hy-tmp/LIBERO-plus')/c['init_state_file']); states=torch.load(init,map_location='cpu',weights_only=False); st=states[int(c.get('init_state_index',0))]; cr=res[c['case_id']]; seeds=cr['seeds']
        env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        try:
            env.seed(seeds['environment_seed']); env.reset(); obs=env.set_init_state(st)
            for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
            obs=env._get_observations()
            if c['factor']=='sensor_noise':
                noise=int(c['runtime_params']['noise']); obs=sensor_transform(obs,noise,seeds['sensor_noise_seed'],0,noise_wrapper)
            expected=cr['initial_observation']['agentview']['sha256']; actual=array_hash(np.asarray(obs['agentview_image'])); assert actual==expected, (c['case_id'],actual,expected)
            lang=BDDLUtils.get_problem_info(bddl)['language_instruction']; po=policy_observation(obs,lang,policy); _,rep=visual_from_obs(policy,po)
            av=po['images']['agentview'][0].detach().cpu().numpy().astype(np.uint8); eye=po['images']['eye_in_hand'][0].detach().cpu().numpy().astype(np.uint8); ga=cv2.cvtColor(av,cv2.COLOR_RGB2GRAY); ge=cv2.cvtColor(eye,cv2.COLOR_RGB2GRAY)
            lap_a=float(cv2.Laplacian(ga,cv2.CV_64F).var()); lap_e=float(cv2.Laplacian(ge,cv2.CV_64F).var()); hf_a=hf_ratio(ga); hf_e=hf_ratio(ge)
            acts=[]
            for j in range(args.m):
                g=torch.Generator(device='cuda:0'); g.manual_seed(seed_for(c['case_id'],j,'diag_z0')); z=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=g); a=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].float().cpu().numpy(); acts.append(np.clip(a,-1,1))
            A=np.stack(acts); grip=A[...,6]
            g=torch.Generator(device='cuda:0'); g.manual_seed(42); zref=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=g); a0=policy.normalizer.unnormalize_action(sample_z(policy,rep,zref))[0].float().cpu().numpy(); a0=np.clip(a0,-1,1)
            pert=[]
            for dx,dy in [(0,2),(0,-2),(2,0),(-2,0)]:
                M=np.float32([[1,0,dx],[0,1,dy]]); arr=cv2.warpAffine(av.astype(np.uint8),M,(av.shape[1],av.shape[0]),borderMode=cv2.BORDER_REPLICATE); po2=dict(po); po2['images']=dict(po['images']); po2['images']['agentview']=torch.tensor(arr[None],device='cuda:0',dtype=torch.uint8); _,r2=visual_from_obs(policy,po2); a2=policy.normalizer.unnormalize_action(sample_z(policy,r2,zref))[0].float().cpu().numpy(); pert.append(np.clip(a2,-1,1))
            P=np.stack(pert); pstd=float(np.mean(np.std(P,axis=0)))
            den=cv2.GaussianBlur(av,(3,3),0); po3=dict(po); po3['images']=dict(po['images']); po3['images']['agentview']=torch.tensor(den[None],device='cuda:0',dtype=torch.uint8); _,r3=visual_from_obs(policy,po3); a3=policy.normalizer.unnormalize_action(sample_z(policy,r3,zref))[0].float().cpu().numpy(); adiff=float(np.mean(np.linalg.norm(np.clip(a3,-1,1)-a0,axis=-1)))
            rows.append({'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'sub_dimension':c.get('sub_dimension'),'success':succ.get(c['case_id']),'lap_agent':lap_a,'lap_eye':lap_e,'hf_agent':hf_a,'hf_eye':hf_e,'s1_xyz_std':float(np.mean(np.std(A[...,:3],axis=0))),'s1_rot_std':float(np.mean(np.std(A[...,3:6],axis=0))),'s1_grip_flip':float(np.mean([np.mean(np.sign(grip[i])!=np.sign(grip[j])) for i in range(len(grip)) for j in range(i+1,len(grip))])),'s2_pert_std':pstd,'s2_pert_xyz_std':float(np.mean(np.std(P[...,:3],axis=0))),'s2_pert_rot_std':float(np.mean(np.std(P[...,3:6],axis=0))),'s4_denoise_action_diff':adiff})
            print(json.dumps({'case':c['case_id'],'factor':c['factor'],'success':rows[-1]['success'],'lap':lap_a,'s1xyz':rows[-1]['s1_xyz_std'],'s2':pstd}),flush=True)
        finally: env.close()
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps({'schema':'factor_signal_scan_v1','rows':rows},indent=2)+'\n')
    print('DONE',len(rows))
if __name__=='__main__': main()
