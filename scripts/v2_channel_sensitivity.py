"""Source-only DINOv2 channel sensitivity on source_fit Clean frames."""
import argparse,json,sys,os,hashlib
from pathlib import Path
import numpy as np, torch, cv2
ROOT=Path('/hy-tmp/patch-diag'); sys.path[:0]=[str(ROOT),'/hy-tmp/LIBERO-plus',str(ROOT/'repos/CoRP')]
os.environ.setdefault('MUJOCO_GL','osmesa'); os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')
from corp import load_policy  # noqa
from corp.preprocessing import prepare_observation  # noqa

def perturb(im,kind):
 x=im.astype(np.float32)
 if kind=='bright_lo': x=x*0.8
 elif kind=='bright_hi': x=x*1.2
 elif kind=='contrast_lo': x=127.5+(x-127.5)*0.8
 elif kind=='contrast_hi': x=127.5+(x-127.5)*1.2
 elif kind=='blur': return cv2.GaussianBlur(im,(3,3),0)
 elif kind=='motion': return cv2.filter2D(im,-1,np.array([[1,1,1],[1,1,1],[1,1,1]],np.float32)/9)
 return np.clip(x,0,255).astype(np.uint8)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--pairs',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
 policy=load_policy(args.checkpoint,device='cuda:0'); policy.eval(); ex=policy.representation_extractor; cfg=policy.config; norm=policy.normalizer
 z=np.load(args.pairs,allow_pickle=False); imgs=z['clean_agentview']; states=z['state8']; n=len(imgs); assert n==len(states)
 kinds=['clean','bright_lo','bright_hi','contrast_lo','contrast_hi','blur','motion']; P={}
 for kind in kinds:
  arr=np.stack([perturb(im,kind) if kind!='clean' else im for im in imgs])
  obs={'images':{'agentview':arr,'eye_in_hand':np.zeros_like(arr)},'state':states,'instruction':['x']*n}
  prep=prepare_observation(obs,config=cfg,normalizer=norm,device=torch.device('cuda:0'))
  with torch.no_grad(): patches=ex.visual_encoder(prep['images']['agentview'])  # [N,256,384]
  P[kind]=patches.float().cpu().numpy()
 base=P['clean']; dom=np.stack([P[k] for k in kinds if k!='clean']); s_dom=dom.std(axis=0).mean(axis=(0,1)); s_task=base.std(axis=0).mean(axis=0); R=s_dom/(s_task+1e-6)
 order=np.argsort(-R); top=order[:max(1,int(.2*384))].tolist()
 out={'schema':'v2_channel_sensitivity_source_v1','pairs':args.pairs,'n_images':n,'images_kind':'source_fit Clean agentview from rgb_oracle_pairs_fresh.npz','perturbations':kinds,'s_domain':s_dom.tolist(),'s_task':s_task.tolist(),'R':R.tolist(),'top20pct':top,'target_dev_images_read':False}
 Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(out,indent=2)+'\n'); print('top',top[:10],'R range',float(R.min()),float(R.max()))
if __name__=='__main__': main()
