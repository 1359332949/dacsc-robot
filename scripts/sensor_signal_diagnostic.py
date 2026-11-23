"""Zero-training sensor-risk signals on frozen dev manifest (t=0)."""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, cv2
ROOT=Path('/hy-tmp/patch-diag')
sys.path[:0]=[str(ROOT),'/hy-tmp/LIBERO-plus',str(ROOT/'repos/CoRP'),str(ROOT/'repos/CoRP/scripts')]
os.environ.setdefault('MUJOCO_GL','osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,policy_observation  # noqa
from corp import load_policy  # noqa
from target_only_action_validation import sample_z, visual_from_obs  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa
from robosuite.utils.binding_utils import MjSimState  # noqa


def load_init_states(path):
    p=Path(path)
    return torch.load(p,map_location='cpu',weights_only=False)

def auroc(pos,neg):
    # higher value -> positive; rank-based
    vals=np.r_[pos,neg]; order=np.argsort(vals); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(vals)+1)
    # average ties
    uniq,inv,cnt=np.unique(vals,return_inverse=True,return_counts=True)
    sums=np.zeros(len(uniq)); np.add.at(sums,inv,ranks); avg=sums/cnt; ranks=avg[inv]
    rp=ranks[:len(pos)].sum(); return float((rp-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))

def rank_auc(pos,neg):
    return auroc(np.asarray(pos),np.asarray(neg))

def at_clean_fpr(score,clean_scores,max_fpr=0.1):
    # choose threshold so clean false positives <= floor(fpr*n)
    n=len(clean_scores); k=max(1,int(np.floor(max_fpr*n)))
    # sort clean ascending; threshold just above kth largest clean if high=risk
    s=sorted(clean_scores); thr=s[-k] if k>0 else max(s)
    return float(thr)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--verified',required=True); ap.add_argument('--out',required=True); ap.add_argument('--m',type=int,default=8); ap.add_argument('--k',type=int,default=4); ap.add_argument('--render-gpu',type=int,default=0); args=ap.parse_args()
    manifest=json.load(open(args.manifest)); verified=json.load(open(args.verified)); succ={r['case_id']:bool(r['success']) for r in verified['results']}
    cases=[c for c in manifest['cases'] if c['factor'] in ('sensor_noise','clean')]
    policy=load_policy(args.checkpoint,device=torch.device('cuda:0')); policy.eval()
    rows=[]
    for c in cases:
        bddl=str(Path('/hy-tmp/LIBERO-plus')/c['bddl']); init=str(Path('/hy-tmp/LIBERO-plus')/c['init_state_file'])
        states=load_init_states(init); idx=int(c.get('init_state_index',0)); st=states[idx] if isinstance(states,(list,tuple)) else states
        env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256,render_gpu_device_id=args.render_gpu)
        try:
            env.seed(7777); env.reset(); sim=env.env.sim; st=np.asarray(st).reshape(-1); nq=int(sim.model.nq); nv=int(sim.model.nv); sim.data.time=float(st[0]); sim.data.qpos[:]=st[1:1+nq]; sim.data.qvel[:]=st[1+nq:1+nq+nv]; sim.forward();
            for _ in range(SETTLING_STEPS): obs,_,_,_=env.step(DUMMY_ACTION)
            obs=env._get_observations()
            po=policy_observation(obs,'',policy) if False else None
            # language from bddl problem info
            import libero.libero.envs.bddl_utils as BDDLUtils
            lang=BDDLUtils.get_problem_info(bddl)['language_instruction']
            po=policy_observation(obs,lang,policy)
            _,rep=visual_from_obs(policy,po)
            # signal 1: multi-z0 action dispersion
            acts=[]
            for j in range(args.m):
                g=torch.Generator(device='cuda:0'); g.manual_seed(700000+j*1009+hash(c['case_id'])%100000)
                z=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=g)
                a=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].float().cpu().numpy(); acts.append(np.clip(a,-1,1))
            A=np.stack(acts); grip=A[...,6]
            img=po['images']['agentview'][0].detach().cpu().numpy() if torch.is_tensor(po['images']['agentview']) else np.asarray(po['images']['agentview'][0])
            # signal 4: clean-vs-denoised action diff
            img_u8=img.astype(np.uint8); den=cv2.GaussianBlur(img_u8,(3,3),0)
            po2=dict(po); po2['images']=dict(po['images']); po2['images']['agentview']=torch.tensor(den[None],device='cuda:0',dtype=torch.uint8)
            _,rep2=visual_from_obs(policy,po2); g=torch.Generator(device='cuda:0'); g.manual_seed(42); zref=torch.randn((1,12,7),device='cuda:0',dtype=torch.float32,generator=g)
            a0=policy.normalizer.unnormalize_action(sample_z(policy,rep,zref))[0].float().cpu().numpy(); a1=policy.normalizer.unnormalize_action(sample_z(policy,rep2,zref))[0].float().cpu().numpy(); adiff=float(np.mean(np.linalg.norm(np.clip(a1,-1,1)-np.clip(a0,-1,1),axis=-1)))
            # signal 2: mild translations of agentview, fixed z0
            pert=[]
            for dx,dy in [(0,2),(0,-2),(2,0),(-2,0)]:
                arr=np.roll(np.roll(img_u8,dx,axis=1),dy,axis=0); po3=dict(po); po3['images']=dict(po['images']); po3['images']['agentview']=torch.tensor(arr[None],device='cuda:0',dtype=torch.uint8)
                _,r3=visual_from_obs(policy,po3); a3=policy.normalizer.unnormalize_action(sample_z(policy,r3,zref))[0].float().cpu().numpy(); pert.append(np.clip(a3,-1,1))
            P=np.stack(pert); pstd=float(np.mean(np.std(P,axis=0)))
            rows.append({'case_id':c['case_id'],'factor':c['factor'],'base_task':c['base_task'],'sub_dimension':c.get('sub_dimension'),'success':succ.get(c['case_id']),'s1_xyz_std':float(np.mean(np.std(A[...,:3],axis=0))),'s1_rot_std':float(np.mean(np.std(A[...,3:6],axis=0))),'s1_grip_flip':float(np.mean([np.mean(np.sign(grip[i])!=np.sign(grip[j])) for i in range(len(grip)) for j in range(i+1,len(grip))])),'s2_pert_std':pstd,'s4_denoise_action_diff':adiff})
            print(json.dumps({k:rows[-1][k] for k in ('case_id','factor','success','s1_xyz_std','s1_grip_flip','s2_pert_std','s4_denoise_action_diff')}),flush=True)
        finally:
            env.close()
    sensor=[r for r in rows if r['factor']=='sensor_noise']; clean=[r for r in rows if r['factor']=='clean']
    summary={}
    for s in ('s1_xyz_std','s1_rot_std','s1_grip_flip','s2_pert_std','s4_denoise_action_diff'):
        pos=[r[s] for r in sensor if r['success'] is False]; neg=[r[s] for r in sensor if r['success'] is True]; cl=[r[s] for r in clean]
        auc=rank_auc(pos,neg); thr=at_clean_fpr(None,cl); rec=float(np.mean([v>=thr for v in pos])) if pos else None; fpr=float(np.mean([v>=thr for v in cl]))
        summary[s]={'auc_fail_vs_success':auc,'threshold_clean_fpr10':thr,'recall_at_clean_fpr10':rec,'clean_fpr_actual':fpr,'clean_n':len(cl),'sensor_success_n':len(neg),'sensor_fail_n':len(pos)}
    out={'schema':'sensor_signal_diagnostic_v1','m':args.m,'k':args.k,'rows':rows,'summary':summary}
    Path(args.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
