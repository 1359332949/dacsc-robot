"""Paired Dev-only EGL rollouts of the fixed block-3 operator from 4e850b8."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORP = ROOT / 'repos/CoRP'
LIB = Path('/hy-tmp/LIBERO-plus')
sys.path[:0] = [str(ROOT), str(CORP), str(CORP/'scripts'), str(LIB)]
from dacsc.static_visual_v2 import array_hash, authorize, file_hash, seed_for, sensor_transform
from dacsc.single_block_probe import VIEWS, align

KEYS = ('agentview_image', 'robot0_eye_in_hand_image')
ARMS = [('frozen', 0.), ('dead050', .50), ('dead075', .75)]
MANIFEST = 'd3e0d8c642cd215c879d2f0174663ff5ec585cd5c6d6b531a213c26e9fb982c0'
CHECKPOINT = 'f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6'
SOURCE = 'fbb03322881190546872010a3a94684991a7eab2f9b3f95943bfdce5ceac9e27'
REFERENCE = '9dceecd7ac5d343b49f2c2c7808becbd50ec6ed90b8077b7bd54c57e66f0f44f'


def observed(obs, code, ns, step, bilateral, wrapper):
    raw = sensor_transform(obs, code, ns, step, wrapper)
    if bilateral:
        assert code > 0
        wrist = sensor_transform(dict(obs, agentview_image=obs[KEYS[1]]), code, seed_for(ns,'wrist'), step, wrapper)
        raw[KEYS[1]] = wrist[KEYS[0]]
    return raw


def match_pixels(raw, key, cache, audit):
    if key not in cache:
        cache[key] = tuple(np.asarray(raw[k]).copy() for k in KEYS)
        return raw
    raw = dict(raw)
    for k, ref in zip(KEYS, cache[key]):
        actual = np.asarray(raw[k]); assert actual.dtype == ref.dtype and actual.shape == ref.shape
        if not np.array_equal(actual, ref):
            delta = np.abs(actual.astype(float) - ref.astype(float)); n = int((delta.max(-1)>0).sum()); mx = float(delta.max())
            assert n <= 64 and mx <= 1, (key, k, n, mx, 'not rounding')
            audit.append([key[0], k, n, mx]); raw[k] = ref
    return raw


def main():
    import torch
    from corp import load_policy
    from target_only_action_validation import sample_z,visual_from_obs
    from evaluate_libero import DUMMY_ACTION,SETTLING_STEPS,is_success,policy_observation
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.benchmark import grab_language_from_filename
    import libero.libero.envs.env_wrapper as wrapper
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=1);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--pilot',action='store_true');ap.add_argument('--resume-complete',action='store_true');a=ap.parse_args()
    assert os.environ['MUJOCO_GL']==os.environ['PYOPENGL_PLATFORM']=='egl' and 0<=a.shard<a.workers
    final=ROOT/'docs/data/static_visual_v2_suite/final';mp=final/'dev_manifest.json';assert file_hash(mp)==MANIFEST
    m=json.loads(mp.read_text());cases=sorted(m['cases'],key=lambda c:c['case_id']);assert m['kind']=='dev' and len(cases)==100
    for c in cases:authorize(m,c,'paired block 3 EGL rollout',MANIFEST,mp)
    inputs=ROOT/'runs/invariance_sensitivity_v1b';assert file_hash(inputs/'sources.json')==SOURCE and file_hash(inputs/'references.npz')==REFERENCE
    clean={c['base_task']:c['case_id'] for c in cases if c['factor']=='clean'};folds=json.loads((inputs/'sources.json').read_text());assert len(clean)==10 and set(folds)==set(clean)
    with np.load(inputs/'references.npz',allow_pickle=False) as z:
        refs={t:tuple(torch.as_tensor(z[f'{t}__3__{p}'],device='cuda:0') for p in ('mean','std','tau')) for t in clean}
    for t in clean:assert len(folds[t])==9 and set(folds[t])==set(clean.values())-{clean[t]} and all(x.shape==(2,384) and x.dtype==torch.float32 for x in refs[t])
    cp=CORP/'checkpoints/corp_libero.pt';assert file_hash(cp)==CHECKPOINT
    policy=load_policy(cp,device='cuda:0').eval().requires_grad_(False);ext=policy.representation_extractor;assert tuple(ext.view_names)==VIEWS and len(ext.visual_encoder.backbone.blocks)==12
    jobs=[(c,b,s) for s in (0,1,2) for c in cases for b in ([False,True] if c['factor']=='sensor_noise' else [False])];assert len(jobs)==390
    if a.pilot:jobs=[next(j for j in jobs if j[1])]
    a.out.mkdir(parents=True,exist_ok=True)
    for c,b,s in jobs[a.shard::a.workers]:
        cid=c['case_id'];task=c['base_task'];regime='dual' if b else 'standard';seeds=json.loads((final/'results'/(cid+'.json')).read_text())['seeds'];es=seeds['environment_seed'];ns=seeds['sensor_noise_seed'];ps=seeds['policy_flow_seed'] if not s else seed_for(c['pair_group_id'],s,'egl_policy_seed');lang=grab_language_from_filename(c['suite'],task+'.bddl');code=int(c['runtime_params']['noise']);assert file_hash(LIB/c['bddl'])==c['resolved_bddl_hash'] and file_hash(LIB/c['init_state_file'])==c['init_state_file_sha256'];initial=torch.load(LIB/c['init_state_file'],weights_only=False,map_location='cpu')[c['init_state_index']];assert array_hash(np.asarray(initial))==c['raw_init_hash']
        dest=a.out/f'{regime}_{s}_{cid}.json'
        if a.resume_complete and dest.exists():
            saved=json.loads(dest.read_text());assert (saved['case_id'],saved['regime'],saved['seed'],saved['reference'])==(cid,regime,s,REFERENCE) and [x['method'] for x in saved['arms']]==[k for k,_ in ARMS];continue
        assert not dest.exists();rows=[];cache={}
        for name,alpha in ARMS:
            start=time.monotonic();random.seed(es);np.random.seed(es);torch.manual_seed(es);ctypes.CDLL(None).srand(ctypes.c_uint(es));env=OffScreenRenderEnv(bddl_file_name=str(LIB/c['bddl']),camera_heights=256,camera_widths=256,render_gpu_device_id=0);hook=None;calls=0
            def fix(module,args,x):
                nonlocal calls
                assert x.shape[0]==2;calls+=1;return align(x,refs[task],alpha,'dead')
            try:
                env.noise=0;env.seed(es);env.reset();obs=env.set_init_state(initial)
                for _ in range(SETTLING_STEPS):obs=env.step(DUMMY_ACTION)[0]
                if alpha:hook=ext.visual_encoder.backbone.blocks[2].register_forward_hook(fix)
                g=torch.Generator(device='cuda:0').manual_seed(ps);steps=0;trace=[];audit=[];h=hashlib.sha256()
                with torch.no_grad():
                    while steps<220:
                        raw=observed(obs,code,ns,steps,b,wrapper);key=(steps,array_hash(env.sim.get_state().flatten()),*(array_hash(np.asarray(raw[k])) for k in ('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')));raw=match_pixels(raw,key,cache,audit);po=policy_observation(raw,lang,policy);z=torch.randn((1,12,7),device='cuda:0',generator=g);_,rep=visual_from_obs(policy,po);plan=policy.normalizer.unnormalize_action(sample_z(policy,rep,z))[0].clamp(-1,1).cpu().numpy();assert plan.shape==(12,7) and np.isfinite(plan).all()
                        trace.append(dict(step=steps,state=key[1],images=[array_hash(np.asarray(raw[k])) for k in KEYS],z=array_hash(z.cpu().numpy()),plan=array_hash(plan)))
                        for action in plan:
                            h.update(action.tobytes());obs,rew,done,info=env.step(action);steps+=1;success=bool(is_success(env,info,rew,done))
                            if done or success or steps>=220:break
                        if done or success:break
                assert calls==(len(trace) if alpha else 0)
                if rows:
                    f=rows[0]['trace'];assert all(trace[0][k]==f[0][k] for k in ('images','state','z'));assert all(x['z']==y['z'] for x,y in zip(f,trace))
                rows.append(dict(method=name,success=success,steps=steps,chunks=len(trace),hook_calls=calls,trace=trace,roundings=audit,action_sha=h.hexdigest(),seconds=time.monotonic()-start))
            finally:
                if hook is not None:hook.remove()
                env.close()
        tmp=dest.with_suffix('.pending');tmp.write_text(json.dumps(dict(case_id=cid,task=task,regime=regime,seed=s,factor=c['factor'],sub=c['sub_dimension'],reference=REFERENCE,env_seed=es,sensor_seed=ns,policy_seed=ps,arms=rows),allow_nan=False));tmp.rename(dest);print(dest.name,flush=True)

if __name__=='__main__':main()
