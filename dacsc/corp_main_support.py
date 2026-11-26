"""Pinned CoRP inference and reset helpers for the fair five-arm table."""
import ctypes,hashlib,json,os,random,sys
from pathlib import Path
import numpy as np
import torch

REPO=Path('/hy-tmp/patch-diag');LIB=Path('/hy-tmp/LIBERO-plus');CORP=REPO/'repos/CoRP'
sys.path[:0]=[str(CORP),str(CORP/'scripts'),str(LIB)]
from corp import load_policy
from corp.preprocessing import prepare_observation
from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import grab_language_from_filename
from dacsc.static_visual_v2 import array_hash,seed_for,sensor_transform
from dacsc.single_block_probe import align

FROZEN_REF='f5aa659cd4b4fe7e02a512bb1f884ebf19a6711c87f3c62831ddca004b11b218'
CHECKPOINT='f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6'
ARMS=('frozen','pixel_align','deep_block12','actmad_inspired_viewwise_v2','ours_v1')
KEYS=('agentview_image','robot0_eye_in_hand_image');VIEWS=('agentview','eye_in_hand')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);assert not p.exists(),p
    t=p.with_suffix('.pending');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.rename(p)
def write(p,obj):
    p=Path(p);t=p.with_suffix('.pending');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def load():
    assert os.environ['MUJOCO_GL']==os.environ['PYOPENGL_PLATFORM']=='egl'
    assert sha(CORP/'checkpoints/corp_libero.pt')==CHECKPOINT
    p=load_policy(CORP/'checkpoints/corp_libero.pt',device='cuda:0').eval().requires_grad_(False)
    assert all(x.dtype==torch.float32 for x in p.parameters())
    assert tuple(p.representation_extractor.view_names)==VIEWS
    assert len(p.representation_extractor.visual_encoder.backbone.blocks)==12
    assert p.action_generator.action_horizon==12
    assert p.action_generator.num_sample_steps==10  # E=12 is execution, not Euler iterations.
    return p
@torch.no_grad()
def predict(policy,raw,z):
    b=prepare_observation(raw,config=policy.config,normalizer=policy.normalizer,device='cuda:0')
    rep=policy.representation_extractor(**b);gen=policy.action_generator
    action=z.to(device=rep.tokens.device,dtype=rep.action_dtype).clone();prefix=gen.prepare_prefix(rep)
    steps=gen.num_sample_steps
    for step in range(steps):
        tau=torch.full((action.shape[0],),step/float(steps),device=action.device,dtype=action.dtype)
        action=action+gen(action,tau,prefix)*(1./steps)
    return policy.normalizer.unnormalize_action(action).clamp(-1,1)
def setup(c,seed):
    seeds=c['seeds'][str(seed)];es=seeds['environment']
    random.seed(es);np.random.seed(es);torch.manual_seed(es);ctypes.CDLL(None).srand(ctypes.c_uint(es))
    env=OffScreenRenderEnv(bddl_file_name=str(LIB/c['bddl']),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
    env.noise=0;env.seed(es);env.reset()
    states=torch.load(LIB/c['init_file'],map_location='cpu',weights_only=False);initial=states[c['init_index']]
    assert array_hash(np.asarray(initial))==c['raw_init_hash']
    obs=env.set_init_state(initial)
    for _ in range(SETTLING_STEPS):obs=env.step(DUMMY_ACTION)[0]
    lang=grab_language_from_filename(c['suite'],c['task']+'.bddl')
    assert lang==c['language'] and not any(x.isdigit() for x in lang)
    return env,obs,lang,torch.Generator(device='cuda:0').manual_seed(seeds['policy'])
def physical_key(env,obs,step):
    return (step,array_hash(env.sim.get_state().flatten()),*(array_hash(np.asarray(obs[k])) for k in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')))
def observed(obs,c,seed,step,wrapper):
    if not c['noise']:return obs
    assert c['factor']=='sensor_bilateral' and c['bilateral']
    ns=c['seeds'][str(seed)]['sensor'];out=sensor_transform(obs,c['noise'],ns,step,wrapper)
    wrist=sensor_transform(dict(obs,agentview_image=obs[KEYS[1]]),c['noise'],seed_for(ns,'wrist'),step,wrapper)
    out[KEYS[1]]=wrist[KEYS[0]];return out
def match_pixels(raw,key,cache,audit):
    if key not in cache:cache[key]=tuple(np.asarray(raw[k]).copy() for k in KEYS);return raw
    out=dict(raw)
    for k,ref in zip(KEYS,cache[key]):
        actual=np.asarray(raw[k]);assert actual.shape==ref.shape and actual.dtype==ref.dtype
        if not np.array_equal(actual,ref):
            d=np.abs(actual.astype(float)-ref.astype(float));n=int((d.max(-1)>0).sum());mx=float(d.max())
            assert n<=64 and mx<=1,(key,k,n,mx,'beyond original EGL rounding allowance')
            audit.append([key[0],k,n,mx]);out[k]=ref
    return out
class Calibration:
    def __init__(self,path,task=None):
        with np.load(path,allow_pickle=False) as z:
            assert str(z['schema'].item())=='corp_fair_5arm_v2'
            names=list(z['target_tasks']);prefix=f'task{names.index(task)}_' if task is not None else 'global_'
            self.mu,self.ss,self.tau=[torch.as_tensor(z[prefix+k],device='cuda:0',dtype=torch.float32) for k in ('mean','std','tau')]
            self.pixel_mean,self.pixel_std=[torch.as_tensor(z[prefix+'pixel_'+k],device='cuda:0',dtype=torch.float32) for k in ('mean','std')]
        assert self.mu.shape==self.ss.shape==self.tau.shape==(12,2,384)
        assert all(torch.isfinite(x).all() for x in (self.mu,self.ss,self.tau,self.pixel_mean,self.pixel_std))
        assert self.pixel_mean.shape==self.pixel_std.shape==(2,3)
        assert ((self.pixel_mean>=.05)&(self.pixel_mean<=.95)).all()
        assert ((self.pixel_std>=.05)&(self.pixel_std<=.50)).all()
    def ref(self,block_idx):
        assert 0<=block_idx<12
        return self.mu[block_idx],self.ss[block_idx],self.tau[block_idx]
