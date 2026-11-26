"""Route A: fixed FairPhysical100, five paired arms, atomic complete jobs.

--workers/--shard partition (case,seed) jobs. --preflight runs no full evaluation.
Completed jobs are validated before resume; incomplete jobs rerun all five arms.
"""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dacsc.corp_main_support import *
from dacsc.baselines_tta import pixel_align_raw,ActMADInspired,viewwise_patch_moments
from dacsc.plus_glass_exact import glass_blur_exact
from unittest.mock import patch
from types import SimpleNamespace


def context(root,suite):
    m=json.loads((root/(suite+'_manifest.json')).read_text());r=json.loads((root/'calibration_receipt.json').read_text())
    assert sha(OLD_REF())==FROZEN_REF and r['frozen_ours_sha256']==FROZEN_REF
    assert r['object_source_frames']==0 and len(r['source_tasks'])==10
    for key in ('loto','global'):assert sha(r[key+'_path'])==r[key+'_sha256']
    assert len(m['cases'])==100 and len({c['case_id'] for c in m['cases']})==100
    assert m['H']==m['E']==12 and m['Euler_steps']==10 and m['max_steps']==220
    if suite=='object':assert set(c['task'] for c in m['cases']).isdisjoint(r['source_tasks'])
    for c in m['cases']:
        assert sha(LIB/c['bddl'])==c['bddl_sha256'] and sha(LIB/c['init_file'])==c['init_sha256']
        assert (c['factor']=='sensor_bilateral')==c['bilateral']==(c['noise']>0)
    return m,r
def OLD_REF():return REPO/'runs/full_traj_v1/full.npz'
def references(r,suite,task):
    cal=Calibration(r['loto_path'],task) if suite=='spatial' else Calibration(r['global_path'])
    if suite=='spatial':
        with np.load(OLD_REF(),allow_pickle=False) as z:ours=tuple(torch.as_tensor(z[task+'__3__'+k],device='cuda:0') for k in ('mean','std','tau'))
        assert all(torch.equal(x,y) for x,y in zip(ours,cal.ref(2)))
    else:ours=cal.ref(2)
    return cal,ours
def noise_wrapper():
    import libero.libero.envs.env_wrapper as wrapper
    # Only this one official evaluator function is accelerated. Preflight verifies
    # pixels and MT19937 state against this environment's own original function.
    return SimpleNamespace(**{k:(glass_blur_exact if k=='glass_blur' else getattr(wrapper,k))
        for k in ('motion_blur','gaussian_blur','zoom_blur','fog','glass_blur')})
def pixel_raw(po,cal):
    # Native prepare_observation moves uint8 inputs to CUDA before division;
    # returning the existing CUDA uint8 output avoids an unnecessary roundtrip.
    return pixel_align_raw(po,cal)
def attach(policy,arm,cal,ours,alpha=.75):
    blocks=policy.representation_extractor.visual_encoder.backbone.blocks
    if arm in ('ours_v1','deep_block12'):
        idx=2 if arm=='ours_v1' else 11;ref=ours if idx==2 else cal.ref(11)
        return blocks[idx].register_forward_hook(lambda mod,args,x:align(x,ref,alpha,'dead')),None
    if arm=='actmad_inspired_viewwise_v2':
        adapter=ActMADInspired(policy,cal)
        assert adapter.implementation_id=='actmad_inspired_viewwise_v2' and adapter.max_frames==4
        return None,adapter
    return None,None
def batch(policy,po):return prepare_observation(po,config=policy.config,normalizer=policy.normalizer,device='cuda:0')
def state_equal(policy,initial):return all(torch.equal(v,initial[k]) for k,v in policy.representation_extractor.visual_encoder.backbone.state_dict().items())


def run_job(policy,c,seed,suite,r,fingerprint):
    cal,ours=references(r,suite,c['task']);rows=[];cache={};frozen_po=None;frozen_z=None;frozen_plan=None
    for arm in ARMS:
        assert all(not p.requires_grad for p in policy.parameters())
        env,obs,lang,g=setup(c,seed);hook=None;adapter=None;roundings=[];trace=[];acthash=hashlib.sha256();start=time.monotonic()
        try:
            hook,adapter=attach(policy,arm,cal,ours);steps=0;success=False;updates=[]
            while steps<220:
                raw=observed(obs,c,seed,steps,noise_wrapper());key=physical_key(env,raw,steps)
                raw=match_pixels(raw,key,cache,roundings);po=policy_observation(raw,lang,policy)
                inp=pixel_raw(po,cal) if arm=='pixel_align' else po
                if adapter is not None:
                    loss=adapter.update(batch(policy,po));assert np.isfinite(loss) and np.isfinite(adapter.last_param_l2)
                    updates.append(dict(frames=len(adapter.frames),loss=loss,parameter_step_l2=adapter.last_param_l2))
                z=torch.randn((1,12,7),device='cuda:0',generator=g);plan=predict(policy,inp,z)[0].cpu().numpy()
                assert plan.shape==(12,7) and np.isfinite(plan).all()
                trace.append(dict(step=steps,state=key[1],robot=list(key[2:]),images=[array_hash(np.asarray(raw[k])) for k in KEYS],
                                  z=array_hash(z.cpu().numpy()),plan=array_hash(plan),eef=np.asarray(raw['robot0_eef_pos']).tolist()))
                if arm=='frozen' and steps==0:frozen_po=po;frozen_z=z.clone();frozen_plan=plan.copy()
                for ac in plan:
                    acthash.update(ac.tobytes());obs,rew,done,info=env.step(ac);steps+=1;success=bool(is_success(env,info,rew,done))
                    if done or success or steps>=220:break
                if done or success:break
            if rows:
                f=rows[0]['trace'];assert all(trace[0][k]==f[0][k] for k in ('state','robot','images','z')),(c['case_id'],arm,'initial mismatch')
                assert all(a['z']==b['z'] for a,b in zip(f,trace)),(c['case_id'],arm,'flow noise mismatch')
            rows.append(dict(arm=arm,success=success,steps=steps,chunks=len(trace),trace=trace,roundings=roundings,
                             finite=True,action_sha256=acthash.hexdigest(),seconds=time.monotonic()-start,updates=updates))
        finally:
            if hook is not None:hook.remove()
            if adapter is not None:
                adapter.close();assert state_equal(policy,adapter.initial)
                assert all(not p.requires_grad for p in policy.parameters())
                restored=predict(policy,frozen_po,frozen_z)[0].cpu().numpy()
                assert np.array_equal(restored,frozen_plan),'adapter close changed Frozen prediction'
                del adapter
            env.close()
    return dict(schema='corp_main_job_v2',run_fingerprint=fingerprint,suite=suite,case_id=c['case_id'],task=c['task'],seed=seed,
        factor=c['factor'],sub_group=c['sub_group'],bilateral=c['bilateral'],language=c['language'],init_index=c['init_index'],
        seeds=c['seeds'][str(seed)],calibration_sha256=r['loto_sha256'] if suite=='spatial' else r['global_sha256'],
        frozen_ours_sha256=FROZEN_REF if suite=='spatial' else r['global_sha256'],arms=rows,
        paired_initial_state_pixels_and_flow=True,all_weights_restored_and_frozen=True)


def preflight(root):
    policy=load();checks={};summaries={}
    # Explicit view-major regression oracle (not a single-frame test).
    for n in (1,2,3,4):
        x=torch.cat([torch.zeros(n,256,384),torch.ones(n,256,384)*100]).cuda()
        mu,_=viewwise_patch_moments(x,n,2);assert torch.equal(mu[0],torch.zeros_like(mu[0])) and torch.equal(mu[1],torch.ones_like(mu[1])*100)
    import libero.libero.envs.env_wrapper as orig
    from PIL import Image
    glass_checks=[]
    for sev in range(1,11):
        pixels=np.random.default_rng(20261126+sev).integers(0,256,size=(256,256,3),dtype=np.uint8);image=Image.fromarray(pixels)
        np.random.seed(123+sev);old=orig.glass_blur(image,sev);st0=np.random.get_state()
        np.random.seed(123+sev);new=glass_blur_exact(image,sev);st1=np.random.get_state()
        assert np.array_equal(old,new) and all(np.array_equal(a,b) for a,b in zip(st0,st1));glass_checks.append(sev)
    for suite in ('spatial','object'):
        m,r=context(root,suite);checks[suite]=[]
        for factor in ('clean','sensor_bilateral','lighting','background'):
            c=next(c for c in m['cases'] if c['factor']==factor);cal,ours=references(r,suite,c['task'])
            env,obs,lang,g=setup(c,0)
            try:
                raw=observed(obs,c,0,0,noise_wrapper());po=policy_observation(raw,lang,policy)
                size=policy.config['image_size']
                assert all(tuple(po['images'][v].shape)==(1,size,size,3) for v in VIEWS)
                z=torch.randn((1,12,7),device='cuda:0',generator=g);base=predict(policy,po,z)
                with patch('torch.randn',side_effect=lambda *a,**kw:z.clone()) as draw:
                    native=policy.predict(po,num_steps=10).clamp(-1,1)
                    assert draw.call_count==1
                assert torch.equal(base,native),'explicit-noise path differs from native policy.predict'
                for arm in ('ours_v1','deep_block12'):
                    h,_=attach(policy,arm,cal,ours,alpha=0)
                    try:zero=predict(policy,po,z)
                    finally:h.remove()
                    assert torch.equal(base,zero)
                if factor=='clean':
                    adapter=ActMADInspired(policy,cal);b=batch(policy,po);losses=[]
                    try:
                        assert torch.equal(predict(policy,po,z),base)
                        for _ in range(4):
                            loss=adapter.update(b);assert np.isfinite(loss) and adapter.last_param_l2>0;losses.append(loss)
                        delta=float((predict(policy,po,z)-base).norm());assert delta>0
                        adapter.reset();assert state_equal(policy,adapter.initial) and torch.equal(predict(policy,po,z),base)
                    finally:adapter.close()
                    assert state_equal(policy,adapter.initial) and torch.equal(predict(policy,po,z),base)
                    assert all(not p.requires_grad and p.grad is None for p in policy.parameters())
                    summaries[suite]=dict(actmad_action_change_l2=delta,four_losses=losses,reset_and_close_exact=True)
                if factor=='sensor_bilateral':
                    assert all(not np.array_equal(raw[k],obs[k]) for k in KEYS),'healthy-eye loophole'
                checks[suite].append(dict(factor=factor,case_id=c['case_id'],native_action_exact=True,alpha0_both_layers_exact=True,
                                           clean_base_language=lang,true_sub_group=c['sub_group']))
            finally:env.close()
    save(root/'preflight.json',dict(status='PASS',suites=checks,adapter=summaries,view_major_frames=[1,2,3,4],
        glass_float_pixels_and_rng_exact_levels=glass_checks,layer_indices=dict(ours=2,deep=11),H=12,E=12,Euler_steps=10))


def fingerprint(root,suite):
    config=json.loads((root/'run_protocol.json').read_text())
    assert sha(LIB/'libero/libero/envs/env_wrapper.py')==config['sensor_wrapper_sha256']
    for p,h in config['source_sha256'].items():assert sha(root/'source_code'/p)==h,p
    assert json.loads((root/'preflight.json').read_text())['status']=='PASS'
    assert sha(root/(suite+'_manifest.json'))==config['manifest_sha256'][suite]
    return config['fingerprints'][suite]


def run(a):
    m,r=context(a.out,a.suite);fp=fingerprint(a.out,a.suite);policy=load()
    jobs=[(c,s) for c in m['cases'] for s in a.seeds];dest=a.out/a.suite/'jobs';dest.mkdir(parents=True,exist_ok=True)
    assert len(jobs)==100*len(a.seeds)
    for c,s in jobs[a.shard::a.workers]:
        p=dest/f"{c['case_id']}__s{s}.json"
        if p.exists():
            old=json.loads(p.read_text());assert old['run_fingerprint']==fp and old['case_id']==c['case_id'] and old['seed']==s
            assert [r['arm'] for r in old['arms']]==list(ARMS) and all(r['finite'] for r in old['arms'])
            continue
        row=run_job(policy,c,s,a.suite,r,fp);save(p,row)
        print('JOB_DONE',a.suite,c['case_id'],s,[(x['arm'],x['success']) for x in row['arms']],flush=True)


def latency(root):
    m,r=context(root,'spatial');c=next(c for c in m['cases'] if c['factor']=='clean');policy=load();cal,ours=references(r,'spatial',c['task'])
    env,obs,lang,g=setup(c,0);po=policy_observation(obs,lang,policy);z=torch.randn((1,12,7),device='cuda:0',generator=g);b=batch(policy,po)
    rows=[]
    try:
        for _ in range(10):predict(policy,po,z)
        for round_idx in range(5):
            order=list(ARMS[round_idx:]+ARMS[:round_idx])
            for arm in order:
                h,adapter=attach(policy,arm,cal,ours)
                try:
                    if adapter:
                        for _ in range(4):adapter.update(b)
                    else:
                        for _ in range(2):predict(policy,pixel_raw(po,cal) if arm=='pixel_align' else po,z)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
                    if adapter:adapter.update(b)
                    action=predict(policy,pixel_raw(po,cal) if arm=='pixel_align' else po,z)
                    torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
                    assert torch.isfinite(action).all()
                    rows.append(dict(arm=arm,round=round_idx,ms=ms,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                                     peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2))
                finally:
                    if h:h.remove()
                    if adapter:adapter.close();del adapter
                    gc.collect()
        med={a:float(np.median([v['ms'] for v in rows if v['arm']==a])) for a in ARMS}
        save(root/'latency.json',dict(rows=rows,median_ms=med,relative={a:med[a]/med['frozen'] for a in ARMS},
            scope='CUDA-synchronized steady replan; ActMAD includes 4-frame SGD+parameter-step diagnostic+prediction, excludes episode construction/prefill/close',
            rounds=5,H=12,E=12,Euler_steps=10,all_arms_episodically_restored=True))
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=('spatial','object'),default='spatial');p.add_argument('--out',required=True,type=Path)
    p.add_argument('--seeds',default='0,1,2');p.add_argument('--workers',type=int,default=8);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--preflight',action='store_true');p.add_argument('--latency',action='store_true');a=p.parse_args();a.seeds=[int(x) for x in a.seeds.split(',')]
    assert a.seeds==[0,1,2] and 0<=a.shard<a.workers
    if a.preflight:preflight(a.out)
    elif a.latency:latency(a.out)
    else:run(a)
