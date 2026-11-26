"""Instrumentation around unchanged upstream SmolVLA observation/action and suite IO."""
from __future__ import annotations
import ctypes, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from roboeval.policies.smolvla import SmolVLAPolicy
from roboeval.suites.libero_spatial import LiberoSpatialSuite
from roboeval.suites.base import EpisodeSpec
from dacsc.smolvla_v1_hook import LAYERS, SmolVLAV1Hook

REVISION='6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
CHECKPOINT_SHA='71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_hash(x):
    x=np.ascontiguousarray(x)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()
def rng_hash():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
def seed_torch(seed):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),path
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.rename(path)


def load_actor():
    actor=SmolVLAPolicy(revision=REVISION)
    actor.load()
    model=actor._model
    model.eval().requires_grad_(False)
    assert model.config.load_vlm_weights is True
    assert model.config.n_action_steps==1 and model.config.num_steps==10 and model.config.chunk_size==50
    assert next(model.model.vlm_with_expert.vlm.model.vision_model.parameters()).dtype==torch.bfloat16
    assert len(model.model.vlm_with_expert.vlm.model.vision_model.encoder.layers)==12
    return actor


class FixedInitSuite(LiberoSpatialSuite):
    def __init__(self, init_index):
        super().__init__(max_steps=220,episodes_per_task=1)
        self.init_index=init_index
    def _build_env(self,bddl):
        env=super()._build_env(bddl)
        env.noise=0
        return env
    def iter_episodes(self,seed=0,n=10):
        bm,_,_=self._import_libero();bench=bm.get_benchmark_dict()['libero_spatial']()
        for spec in super().iter_episodes(seed,n):
            meta=self._spec_index.pop(spec.episode_id)
            states=bench.get_task_init_states(meta['task_index'])
            assert 0<=self.init_index<len(states)
            meta['init_state']=states[self.init_index]
            meta['episode_index']=self.init_index
            eid=f"phase3a-t{meta['task_index']:02d}-i{self.init_index}"
            self._spec_index[eid]=meta
            yield EpisodeSpec(eid,spec.task,spec.seed)


def setup_case(task,init_index):
    suite=FixedInitSuite(init_index)
    specs=list(suite.iter_episodes(seed=0,n=10))
    assert len(specs)==10
    spec=specs[task]
    random.seed(spec.seed);np.random.seed(spec.seed);ctypes.CDLL(None).srand(ctypes.c_uint(spec.seed))
    obs=suite.reset(spec)
    assert int(suite._env.noise)==0
    return suite,spec,obs


def snapshot(suite,obs):
    return {'sim':array_hash(suite._env.sim.get_state().flatten()),
            'agent':array_hash(obs['agentview_image']), 'wrist':array_hash(obs['robot0_eye_in_hand_image']),
            'eef':array_hash(np.asarray(obs['robot0_eef_pos'])),
            'quat':array_hash(np.asarray(obs['robot0_eef_quat'])),
            'gripper':array_hash(np.asarray(obs['robot0_gripper_qpos']))}


def cache_observation(path,obs):
    assert not path.exists()
    np.savez_compressed(path,**{k:obs[k] for k in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','language')})
def read_observation(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:(str(z[k].item()) if k=='language' else z[k].copy()) for k in z.files}


def reference_for(calib,task,device='cuda:0'):
    with np.load(calib,allow_pickle=False) as z:
        assert str(z['schema'].item())=='smolvla_v1_init0_loto_v1'
        return {l:tuple(torch.as_tensor(z[f'task{task}_layer{l}_{k}'],device=device,dtype=torch.float32)
                        for k in ('mu','ss','tau')) for l in LAYERS}


def rollout(actor,task,init_index,policy_seed,mode,refs=None,out=None):
    suite,spec,obs=setup_case(task,init_index)
    start=snapshot(suite,obs)
    initial=obs
    if out is not None:cache_observation(out,obs)
    actor.reset(policy_seed);seed_torch(policy_seed)
    source=mode=='source'
    hook=SmolVLAV1Hook(actor._model,refs,layers=LAYERS if source else (3,),
                      active_layer=3 if mode=='v1' else None,alpha=.75)
    frames=[];means=[];seconds=[];noise=[];ah=hashlib.sha256();began=time.monotonic()
    try:
        for step in range(220):
            noise.append(rng_hash())
            hook.begin();action=actor.act(obs);stats=hook.end()
            assert action.shape==(7,) and np.isfinite(action).all()
            ah.update(action.tobytes())
            if source:
                means.append(torch.stack([torch.stack([stats[l,v]['mean'] for v in range(2)]) for l in LAYERS]).cpu().numpy())
                seconds.append(torch.stack([torch.stack([stats[l,v]['second'] for v in range(2)]) for l in LAYERS]).cpu().numpy())
            else:
                frames.append({'step':step,'active':[float(stats[3,v]['active_fraction']) for v in range(2)],
                               'gain_abs':[float(stats[3,v]['gain_abs_mean']) for v in range(2)],
                               'action':action.tolist(),'eef':np.asarray(obs['robot0_eef_pos']).tolist()})
            result=suite.step(action);obs=result.obs
            if step%40==0:print(json.dumps({'mode':mode,'task':task,'step':step,'elapsed':round(time.monotonic()-began,2)}),flush=True)
            if result.done or step+1>=220:
                success=bool(result.success);break
        row={'task_id':task,'task':spec.task,'init_index':init_index,'policy_seed':policy_seed,
             'environment_seed':spec.seed,'mode':mode,'success':success,'steps':step+1,
             'seconds':time.monotonic()-began,'start':start,'rng_before_replans':noise,
             'actions_sha256':ah.hexdigest(),'frames':frames,'finite_actions':True,
             'two_calls_every_replan':True,'vision_dtype':'torch.bfloat16','all_1024_patches':True}
        if source:
            raw={'mean':np.stack(means),'second':np.stack(seconds)}
            assert all(np.isfinite(x).all() for x in raw.values())
        else:raw=None
        return row,raw,initial
    finally:
        hook.close();suite.close()


def predict_at(actor,obs,refs=None,layer=None,alpha=.75,layers=(3,)):
    actor.reset(0);seed_torch(0)
    hook=SmolVLAV1Hook(actor._model,refs,layers=layers,active_layer=layer,alpha=alpha) if refs is not None else None
    try:
        if hook:hook.begin()
        action=actor.act(obs)
        stats=hook.end() if hook else {}
        assert np.isfinite(action).all()
        return action.copy(),stats
    finally:
        if hook:hook.close()


def scan_layers(actor,obs,refs):
    results=[]
    with torch.random.fork_rng(devices=[0]):
        base,_=predict_at(actor,obs)
        for layer in LAYERS:
            zero,_=predict_at(actor,obs,refs,layer,0,layers=(layer,))
            assert np.array_equal(base,zero),f'alpha0 action drift at layer {layer}'
            action,stats=predict_at(actor,obs,refs,layer,.75,layers=(layer,))
            delta=action-base
            results.append({'layer':layer,'alpha0_action_exact':True,
                'xyz_drift_l2':float(np.linalg.norm(delta[:3])),
                'rotation_drift_l2':float(np.linalg.norm(delta[3:6])),
                'action_drift_l2':float(np.linalg.norm(delta)),
                'active':[float(stats[layer,v]['active_fraction']) for v in range(2)]})
    return results
