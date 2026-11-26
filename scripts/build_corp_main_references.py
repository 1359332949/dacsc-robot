"""Prepare fixed FairPhysical100 manifests; replay the original ten source episodes.

Only Spatial source trajectories are read by fitting. Object is evaluation-only.
"""
import argparse,json,re,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dacsc.corp_main_support import *
from dacsc.static_visual_v2 import resolve_task,authorize

OLD=REPO/'runs/full_traj_v1';FINAL=REPO/'docs/data/static_visual_v2_suite/final'


def prepare(out):
    assert sha(OLD/'full.npz')==FROZEN_REF
    m=json.loads((FINAL/'dev_manifest.json').read_text());assert len(m['cases'])==100
    spatial=[];source=[]
    for c in sorted(m['cases'],key=lambda c:c['case_id']):
        authorize(m,c,'route A FairPhysical development')
        seeds=json.loads((FINAL/'results'/(c['case_id']+'.json')).read_text())['seeds']
        factor='sensor_bilateral' if c['factor']=='sensor_noise' else c['factor']
        row=dict(case_id=c['case_id'],suite='libero_spatial',task=c['base_task'],factor=factor,
            sub_group='clean' if factor=='clean' else c['sub_dimension'],bddl=c['bddl'],bddl_sha256=c['resolved_bddl_hash'],
            init_file=c['init_state_file'],init_sha256=c['init_state_file_sha256'],init_index=c['init_state_index'],
            raw_init_hash=c['raw_init_hash'],noise=int(c['runtime_params']['noise']),bilateral=factor=='sensor_bilateral',
            language=grab_language_from_filename('libero_spatial',c['base_task']+'.bddl'),
            seeds={str(s):dict(environment=seeds['environment_seed'],sensor=seeds['sensor_noise_seed'],
                    policy=seeds['policy_flow_seed'] if s==0 else seed_for(c['pair_group_id'],s,'egl_policy_seed')) for s in (0,1,2)},role='historically_accessed_development')
        spatial.append(row)
        if factor=='clean':source.append(row)
    obj=json.loads((REPO/'docs/data/official_object_physical_manifest.json').read_text());objects=[]
    assert obj['tag']=='official_object_physical_v1'
    for c in obj['cases']:
        if c['factor']=='sensor_noise' and not c['dual']:continue
        fac='sensor_bilateral' if c['factor']=='sensor_noise' else c['factor']
        sub=resolve_task(Path(c['bddl']).stem,'background')['sub_dimension'] if fac=='background' else ('clean' if fac=='clean' else ('light_asset' if fac=='lighting' else ('motion_blur','gaussian_blur','zoom_blur','fog','glass_blur')[(c['noise']-1)//10]))
        states=torch.load(LIB/c['init_file'],map_location='cpu',weights_only=False)
        seeds={str(s):dict(environment=seed_for(obj['tag'],c['task'],c['factor'],c['sub'],c['init_index'],'env',s),
            sensor=seed_for(obj['tag'],c['task'],c['factor'],c['sub'],'sensor',s),
            policy=seed_for(obj['tag'],c['task'],c['factor'],c['sub'],'policy',s)) for s in (0,1,2)}
        objects.append(dict(case_id=c['case_id'],suite='libero_object',task=c['task'],factor=fac,sub_group=sub,
            bddl=c['bddl'],bddl_sha256=sha(LIB/c['bddl']),init_file=c['init_file'],init_sha256=sha(LIB/c['init_file']),
            init_index=c['init_index'],raw_init_hash=array_hash(np.asarray(states[c['init_index']])),noise=c['noise'],
            bilateral=fac=='sensor_bilateral',language=grab_language_from_filename('libero_object',c['task']+'.bddl'),
            seeds=seeds,role='historically_accessed_official_development_retest'))
    for name,cases in [('spatial',spatial),('object',objects)]:
        assert len(cases)==len({c['case_id'] for c in cases})==100
        assert {f:sum(c['factor']==f for c in cases) for f in ('clean','sensor_bilateral','lighting','background')}==dict(clean=10,sensor_bilateral=30,lighting=30,background=30)
        for c in cases:
            assert not any(ch.isdigit() for ch in c['language'])
            assert sha(LIB/c['bddl'])==c['bddl_sha256'] and sha(LIB/c['init_file'])==c['init_sha256']
            assert c['bilateral']==(c['noise']>0)
            assert c['sub_group'] in ('table_texture','scene_texture') if c['factor']=='background' else True
        save(out/(name+'_manifest.json'),dict(schema='corp_main_fairphysical_v2',suite=name,cases=cases,
            seeds=[0,1,2],arms=list(ARMS),H=12,E=12,Euler_steps=10,max_steps=220,renderer='EGL',dtype='FP32',
            frozen_reference_sha256=FROZEN_REF,blind_access=False,source_suite='libero_spatial'))
    save(out/'source_manifest.json',dict(cases=source,frozen_reference_sha256=FROZEN_REF,
         source_original=[dict(path=str(p),sha256=sha(p)) for p in sorted((OLD/'source').glob('*')) if p.is_file()]))


def capture(out,shard,workers):
    cases=json.loads((out/'source_manifest.json').read_text())['cases'];policy=load();blocks=policy.representation_extractor.visual_encoder.backbone.blocks
    for c in cases[shard::workers]:
        dest=out/'source'/(c['case_id']+'.npz');dest.parent.mkdir(exist_ok=True)
        if dest.exists():continue
        legacy=json.loads((OLD/'source'/('standard_0_'+c['case_id']+'.json')).read_text())
        raw_old=np.load(OLD/'source'/(c['case_id']+'.npy'),allow_pickle=False)
        captures={};hooks=[]
        for i,block in enumerate(blocks):
            def make(i):
                def hook(mod,args,x):
                    assert x.shape==(2,257,384) and x.dtype==torch.float32
                    p=x[:,1:].double();captures[i]=torch.stack([p.mean(1),p.square().mean(1)]).cpu().numpy()
                return hook
            hooks.append(block.register_forward_hook(make(i)))
        env,obs,lang,g=setup(c,0);mom=[];pix=[];trace=[];start=time.monotonic();action_hash=__import__('hashlib').sha256()
        try:
            steps=0;success=False
            while steps<220:
                po=policy_observation(obs,lang,policy)
                pix.append(np.stack([np.stack([np.asarray(po['images'][v],dtype=np.float64).mean((0,1,2))/255,
                           np.square(np.asarray(po['images'][v],dtype=np.float64)/255).mean((0,1,2))]) for v in VIEWS]))
                z=torch.randn((1,12,7),device='cuda:0',generator=g);p=predict(policy,po,z)[0].cpu().numpy()
                mom.append(np.stack([captures[i] for i in range(12)],axis=1))
                trace.append(dict(step=steps,state=physical_key(env,obs,steps)[1],images=[array_hash(np.asarray(obs[k])) for k in KEYS],z=array_hash(z.cpu().numpy()),plan=array_hash(p)))
                for ac in p:
                    action_hash.update(ac.tobytes());obs,rew,done,info=env.step(ac);steps+=1;success=bool(is_success(env,info,rew,done))
                    if done or success or steps>=220:break
                if done or success:break
            arr=np.stack(mom);assert arr.shape[1:]==(2,12,2,384)
            b3=arr[:,:,2];exact=b3.shape==raw_old.shape and np.array_equal(b3,raw_old)
            delta=float(np.max(np.abs(b3-raw_old))) if b3.shape==raw_old.shape else None
            oldrow=legacy['arms'][0]
            audit=dict(task=c['task'],case_id=c['case_id'],frames=len(arr),steps=steps,success=success,
                old_frames=len(raw_old),b3_moments_bit_exact=exact,b3_max_abs_error=delta,
                actions_bit_exact=action_hash.hexdigest()==oldrow['action_sha'],trace=trace,old_trace=oldrow['trace'],seconds=time.monotonic()-start)
            assert np.isfinite(arr).all();np.savez_compressed(dest,moments=arr,pixel=np.stack(pix))
            save(dest.with_suffix('.json'),audit);print('SOURCE',c['task'],len(arr),exact,delta,flush=True)
        finally:
            for h in hooks:h.remove()
            env.close()


def fit_all(out):
    cases=json.loads((out/'source_manifest.json').read_text())['cases'];tasks=sorted(c['task'] for c in cases);raw={};pixels={};audits=[]
    for c in cases:
        p=out/'source'/(c['case_id']+'.npz');z=np.load(p,allow_pickle=False)
        raw[c['task']]=torch.from_numpy(z['moments']);pixels[c['task']]=torch.from_numpy(z['pixel'])
        audits.append(json.loads(p.with_suffix('.json').read_text()))
    assert all(a['b3_moments_bit_exact'] for a in audits),'source replay differs from frozen source; inspect before fitting'
    def stats(keys):
        x=torch.cat([raw[t] for t in keys]);m=x[:,0].mean(0);sd=(x[:,1].mean(0)-m.square()).clamp_min(0).sqrt()
        p=torch.cat([pixels[t] for t in keys]);pm=p[:,:,0].mean(0);ps=(p[:,:,1].mean(0)-pm.square()).clamp_min(0).sqrt()
        return m,sd,pm,ps
    def score_hold(t,keys):
        mu,ss,_,_=stats(keys);m=raw[t][:,0];sd=(raw[t][:,1]-m.square()).clamp_min(0).sqrt()
        return ((m-mu).square()+(sd-ss).square())/(ss.square()+1e-6)
    arrays={'schema':np.array('corp_fair_5arm_v2'),'target_tasks':np.array(tasks)};folds={}
    frozen=np.load(OLD/'full.npz',allow_pickle=False);assert sha(OLD/'full.npz')==FROZEN_REF
    for i,t in enumerate(tasks):
        outer=[k for k in tasks if k!=t];mu,ss,pm,ps=stats(outer)
        tau=torch.cat([score_hold(h,[k for k in outer if k!=h]) for h in outer]).quantile(.95,dim=0)
        for j,name in enumerate(('mean','std','tau')):
            recomputed=(mu,ss,tau)[j][2].float().numpy();assert np.array_equal(recomputed,frozen[t+'__3__'+name]),(t,name,'reference mismatch')
        for k,x in dict(mean=mu,std=ss,tau=tau,pixel_mean=pm,pixel_std=ps).items():arrays[f'task{i}_{k}']=x.float().numpy()
        folds[t]=dict(outer=outer,inner={h:[k for k in outer if k!=h] for h in outer},frames=sum(len(raw[k]) for k in outer))
    mu,ss,pm,ps=stats(tasks);tau=torch.cat([score_hold(h,[k for k in tasks if k!=h]) for h in tasks]).quantile(.95,dim=0)
    glob={'schema':arrays['schema'],'target_tasks':arrays['target_tasks']}
    for k,x in dict(mean=mu,std=ss,tau=tau,pixel_mean=pm,pixel_std=ps).items():glob['global_'+k]=x.float().numpy()
    paths=[REPO/'runs/phase2_calib/spatial_loto_5arm_v2.npz',REPO/'runs/phase2_calib/spatial10_zeroshot_5arm_v2.npz']
    for p,data in zip(paths,[arrays,glob]):
        assert not p.exists(),p
        for k,a in data.items():
            if a.dtype.kind in 'f':assert np.isfinite(a).all()
            if 'pixel_mean' in k:assert ((a>=.05)&(a<=.95)).all()
            if 'pixel_std' in k:assert ((a>=.05)&(a<=.50)).all()
        np.savez_compressed(p,**data)
    save(out/'calibration_receipt.json',dict(loto_path=str(paths[0]),loto_sha256=sha(paths[0]),global_path=str(paths[1]),global_sha256=sha(paths[1]),
        frozen_ours_path=str(OLD/'full.npz'),frozen_ours_sha256=FROZEN_REF,folds=folds,source_frames=sum(len(x) for x in raw.values()),
        source_tasks=tasks,object_source_frames=0,source_replay=audits,block3_recomputed_bit_exact=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','source','fit'));p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--shard',type=int,default=0);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.mode=='prepare':prepare(a.out)
    elif a.mode=='source':capture(a.out,a.shard,a.workers)
    else:fit_all(a.out)
