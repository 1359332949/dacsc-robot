"""t=0 visual quality + dual-view asymmetry gate; Clean No-Op vs Sensor OOD."""
from __future__ import annotations
import argparse,json,sys,os
from pathlib import Path
import numpy as np, torch, cv2
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),'/hy-tmp/LIBERO-plus',str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
os.environ.setdefault('MUJOCO_GL','osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation  # noqa
from corp import load_policy  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa
import libero.libero.envs.bddl_utils as BDDLUtils  # noqa
from robosuite.utils.binding_utils import MjSimState  # noqa
from dacsc.static_visual_v2 import sensor_transform, seed_for  # noqa
import libero.libero.envs.env_wrapper as noise_wrapper  # noqa

def state_from_flat(sim,st):
    st=np.asarray(st).reshape(-1); nq=int(sim.model.nq); nv=int(sim.model.nv)
    sim.data.time=float(st[0]); sim.data.qpos[:]=st[1:1+nq]; sim.data.qvel[:]=st[1+nq:1+nq+nv]; sim.forward()

def hf_ratio(gray):
    g=gray.astype(np.float32)/255.0; f=np.fft.fftshift(np.fft.fft2(g)); a=np.abs(f); h,w=a.shape; yy,xx=np.mgrid[0:h,0:w]; r=np.sqrt(((yy-h/2)/(h/2))**2+((xx-w/2)/(w/2))**2); return float(a[r>0.5].mean()/(a.mean()+1e-9))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--verified',required=True); ap.add_argument('--out',required=True); ap.add_argument('--render-gpu',type=int,default=0); args=ap.parse_args()
    man=json.load(open(args.manifest)); ver=json.load(open(args.verified)); succ={r['case_id']:bool(r['success']) for r in ver['results']}
    cases=[c for c in man['cases'] if c['factor'] in ('sensor_noise','clean')]
    policy=load_policy(args.checkpoint,device='cuda:0'); policy.eval(); rows=[]
    for c in cases:
        bddl=str(Path('/hy-tmp/LIBERO-plus')/c['bddl']); init=str(Path('/hy-tmp/LIBERO-plus')/c['init_state_file']); states=torch.load(init,map_location='cpu',weights_only=False); st=states[int(c.get('init_state_index',0))]
        env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        try:
            env.seed(7777); env.reset(); state_from_flat(env.env.sim,st)
            for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
            obs=env._get_observations();
            if c['factor']=='sensor_noise':
                noise=int(c['runtime_params']['noise']); sid=seed_for(2026112301,c['pair_group_id'],0,'sensor_noise_seed'); obs=sensor_transform(obs,noise,sid,0,noise_wrapper)
            lang=BDDLUtils.get_problem_info(bddl)['language_instruction']; po=policy_observation(obs,lang,policy)
            av=po['images']['agentview'][0].detach().cpu().numpy().astype(np.uint8); eye=po['images']['eye_in_hand'][0].detach().cpu().numpy().astype(np.uint8)
            ga=cv2.cvtColor(av,cv2.COLOR_RGB2GRAY); ge=cv2.cvtColor(eye,cv2.COLOR_RGB2GRAY)
            lap_a=float(cv2.Laplacian(ga,cv2.CV_64F).var()); lap_e=float(cv2.Laplacian(ge,cv2.CV_64F).var()); hf_a=hf_ratio(ga); hf_e=hf_ratio(ge)
            rows.append({'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'sub_dimension':c.get('sub_dimension'),'success':succ.get(c['case_id']),'lap_agent':lap_a,'lap_eye':lap_e,'hf_agent':hf_a,'hf_eye':hf_e,'lap_ratio':lap_a/(lap_e+1e-9),'hf_ratio_av_eye':hf_a/(hf_e+1e-9)})
            print(json.dumps({k:rows[-1][k] for k in ('case_id','factor','success','lap_agent','lap_eye','lap_ratio','hf_ratio_av_eye')}),flush=True)
        finally: env.close()
    clean=[r for r in rows if r['factor']=='clean']; sensor=[r for r in rows if r['factor']=='sensor_noise']
    def metrics(vals,cl):
        # higher risk = lower lap_agent or lower hf_agent; use 1/(x+eps)
        scores=[1.0/(v+1e-9) for v in vals]; cs=[1.0/(v+1e-9) for v in cl]; thr=max(cs)
        det=float(np.mean([s>thr for s in scores])); fpr=float(np.mean([s>thr for s in cs])); return det,fpr
    summary={}
    for key in ('lap_agent','hf_agent'):
        vals=[r[key] for r in sensor]; cl=[r[key] for r in clean]; det,fpr=metrics(vals,cl)
        # AUC clean vs sensor (score=low quality -> high risk)
        cs=[1/(v+1e-9) for v in cl]; ss=[1/(v+1e-9) for v in vals]; allv=sorted([(v,1) for v in ss]+[(v,0) for v in cs]); s=sum(1 for v,l in allv if l==1 for w,l2 in allv if l2==0 and w<v)+0.5*sum(1 for v,l in allv if l==1 for w,l2 in allv if l2==0 and w==v); auc=s/(len(ss)*len(cs))
        summary[key]={'clean_vs_sensor_auc':auc,'sensor_detect_at_clean_max':det,'clean_fpr':fpr}
    # subtype summary
    from collections import defaultdict
    sub=defaultdict(list)
    for r in sensor: sub[r.get('sub_dimension')].append(r)
    summary['sensor_subtypes']={k:{'n':len(v),'lap_agent_mean':float(np.mean([x['lap_agent'] for x in v])),'success':sum(x['success'] for x in v)} for k,v in sub.items()}
    Path(args.out).write_text(json.dumps({'schema':'visual_quality_gate_v1','rows':rows,'summary':summary},indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
