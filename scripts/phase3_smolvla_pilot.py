"""Phase3A native-model preflight, paired development Clean guard and isolated timing."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from phase3_smolvla_common import (load_actor,setup_case,predict_at,reference_for,rollout,
    scan_layers,save_json,sha,seed_torch,read_observation,LAYERS,SmolVLAV1Hook)


def preflight(actor,out):
    suite,spec,obs=setup_case(0,0)
    try:
        batch=actor._model.normalize_inputs(actor._build_observation(obs))
        expected,_=actor._model.prepare_images(batch)
        assert len(expected)==2
        vision=actor._model.model.vlm_with_expert.vlm.model.vision_model
        inputs=[]
        def observe(module,args,kwargs):
            x=kwargs['pixel_values'] if 'pixel_values' in kwargs else args[0]
            inputs.append(x.detach())
        root=vision.register_forward_pre_hook(observe,with_kwargs=True)
        capture=SmolVLAV1Hook(actor._model,layers=LAYERS)
        try:
            actor.reset(0);seed_torch(0);capture.begin();base=actor.act(obs);capture.end()
        finally:root.remove();capture.close()
        assert len(inputs)==2
        assert all(torch.equal(inputs[i],expected[i].to(vision.dtype)) for i in range(2))
        assert not torch.equal(inputs[0],inputs[1]),'camera inputs indistinguishable in order check'
        dummy={l:(torch.zeros(2,768,device='cuda'),torch.ones(2,768,device='cuda'),torch.ones(2,768,device='cuda')) for l in LAYERS}
        checks=[]
        for l in LAYERS:
            action,stats=predict_at(actor,obs,dummy,l,0,layers=(l,))
            assert np.array_equal(action,base)
            checks.append(l)
        save_json(out/'preflight.json',{'status':'PASS','layers':list(LAYERS),'two_calls_per_layer':True,
             'shape':[1,1024,768],'no_cls':True,'native_view_order_verified':True,
             'alpha0_full_action_identity_layers':checks,'vision_dtype':str(vision.dtype),
             'load_vlm_weights':actor._model.config.load_vlm_weights,
             'n_action_steps':1,'Euler_steps':10,'policy_seed_control':'external Torch CPU/CUDA reset before each episode'})
    finally:suite.close()


def guard(actor,a):
    cal=a.out/'spatial_init0_calib.npz'
    folds=json.loads((a.out/'folds.json').read_text());assert sha(cal)==folds['calib_sha256']
    dest=a.out/'guard';dest.mkdir(parents=True,exist_ok=True)
    for task in range(a.shard,10,a.workers):
        assert not (dest/f'task{task}.json').exists()
        refs=reference_for(cal,task)
        r0,_,obs=rollout(actor,task,1,0,'frozen',refs,out=dest/f'task{task}_initial.npz')
        r1,_,_=rollout(actor,task,1,0,'v1',refs)
        assert r0['start']==r1['start'],'initial physical state or RGB differed'
        common=min(r0['steps'],r1['steps'])
        assert r0['rng_before_replans'][:common]==r1['rng_before_replans'][:common],'paired model noise streams differed'
        with np.load(cal,allow_pickle=False) as z:assert str(z['task_names'][task])==r0['task']
        scans=scan_layers(actor,read_observation(dest/f'task{task}_initial.npz'),refs)
        save_json(dest/f'task{task}.json',{'rows':[r0,r1],'scan':scans,'start_exact':True,
                   'paired_rng_prefix_exact':True,'calib_sha256':sha(cal),
                   'evaluation_role':'calibration-state-disjoint DEVELOPMENT Clean; init1 previously examined'})
        print('GUARD_DONE',task,r0['success'],r1['success'],flush=True)


def latency(actor,a):
    obs=read_observation(a.out/'guard/task0_initial.npz')
    refs=reference_for(a.out/'spatial_init0_calib.npz',0)
    entries=[]
    def run(arm,measure):
        actor.reset(0);seed_torch(0)
        h=SmolVLAV1Hook(actor._model,refs,layers=(3,),active_layer=3) if arm=='v1' else None
        try:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter()
            if h:h.begin()
            action=actor.act(obs)
            if h:h.end()
            torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
            assert np.isfinite(action).all()
            if measure:entries.append({'arm':arm,'ms':ms,'peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,
                                      'peak_reserved_mb':torch.cuda.max_memory_reserved()/1024**2})
        finally:
            if h:h.close()
    for _ in range(2):
        for arm in ('frozen','v1'):run(arm,False)
    for round_id in range(5):
        for arm in (('frozen','v1') if round_id%2==0 else ('v1','frozen')):
            run(arm,True);entries[-1]['round']=round_id
    med={arm:float(np.median([r['ms'] for r in entries if r['arm']==arm])) for arm in ('frozen','v1')}
    save_json(a.out/'latency.json',{'scope':'end-to-end native act, CUDA synchronized, no environment step; V1 includes diagnostic moment capture',
             'warmup_pairs':2,'interleaved_rounds':5,'rows':entries,'median_ms':med,'relative_median':med['v1']/med['frozen']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('preflight','guard','latency'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    actor=load_actor();versions=[v._version for v in actor._model.parameters()]
    if a.mode=='preflight':preflight(actor,a.out)
    elif a.mode=='guard':guard(actor,a)
    else:latency(actor,a)
    assert versions==[v._version for v in actor._model.parameters()]
    assert all(not p.requires_grad and p.grad is None for p in actor._model.parameters())
