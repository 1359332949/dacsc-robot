"""Frozen Spatial SOH transfer with target-suite Clean calibration, not zero-shot.

Only init0 enters reference fitting. Evaluation uses init1; task T is excluded
from its reference and nested source gate. Spatial SOH/subspace remain fixed.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from roboeval.suites.libero_spatial import LiberoSpatialSuite
from roboeval.suites.base import EpisodeSpec
from phase3_smolvla_common import load_actor, seed_torch, sha, LAYERS, reference_for
from dacsc.smolvla_v1_hook import SmolVLAV1Hook, fit_loto
from dacsc.smolvla_soh import SmolVLASOHAdapter
from phase4_soh_clean5_eval import load_soh, pair_check
import phase3_smolvla_deepgate_closedloop as paired
import phase3_smolvla_shift as shift
import phase3_smolvla_unified_diag as diag

ROOT=Path('/hy-tmp/patch-diag')
OUT=ROOT/'runs/phase4_libero10_transfer_v1'
CALIB=OUT/'libero10_init0_calib.npz'
SOH=ROOT/'runs/phase4_soh/soh_v2_ckpt.pt'
MAX_STEPS=520


def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)


class Libero10Suite(LiberoSpatialSuite):
    name='libero-10'
    def __init__(self,init_index):
        super().__init__(max_steps=MAX_STEPS,episodes_per_task=1)
        self.init_index=init_index
    def _build_env(self,bddl):
        env=super()._build_env(bddl);env.noise=0;return env
    def iter_episodes(self,seed=0,n=10):
        benchmark,get_path,_=self._import_libero();self._patch_torch_load_for_libero()
        bm=benchmark.get_benchmark_dict()['libero_10']()
        assert bm.n_tasks==10 and n==10
        for t in range(10):
            task=bm.get_task(t);states=bm.get_task_init_states(t)
            eid=f'libero10-t{t:02d}-i{self.init_index}'
            self._spec_index[eid]=dict(bddl_file=str(Path(get_path('bddl_files'))/task.problem_folder/task.bddl_file),
                task_index=t,episode_index=self.init_index,language=task.language,init_state=states[self.init_index])
            yield EpisodeSpec(eid,task.name,seed+t)


def setup(case):
    suite=Libero10Suite(case['init_index']);spec=list(suite.iter_episodes())[case['task_id']]
    assert spec.task==case['base_task']
    random.seed(spec.seed);np.random.seed(spec.seed);ctypes.CDLL(None).srand(ctypes.c_uint(spec.seed))
    obs=suite.reset(spec)
    assert suite._env.noise==0 and suite.max_steps==520
    return suite,spec,obs


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    suite=Libero10Suite(0);specs=list(suite.iter_episodes());suite.close()
    cases=[]
    for t,spec in enumerate(specs):
        cases.append(dict(id=f'l10_clean_t{t:02d}_i1',task_id=t,base_task=spec.task,group='Clean',subgroup='clean',
            init_index=1,scored=True,language=suite._spec_index[spec.episode_id]['language']))
    kinds=('motion_blur','gaussian_blur','zoom_blur','glass_blur','fog')
    # Fixed cyclic schedule: five families x four cases; every task appears twice.
    for k,kind in enumerate(kinds):
        for j in range(4):
            t=(4*k+j)%10
            cases.append(dict(cases[t],id=f'l10_sensor_{kind}_t{t:02d}_i1',group='Sensor',subgroup=kind,severity=3))
    protocol=dict(suite='vanilla_libero_10',max_steps=520,warmup=10,H=50,E=1,Euler=10,seed=0,
        precision='native BF16; CPU normalization',renderer='EGL',noise=0,ema_beta=0.,
        calibration='libero_10 init0 Clean; per-task outer9, inner8+1; gate inner7+1',
        evaluation='init1; one policy seed; development transfer pilot, not untouched confirmation',
        transfer='Spatial-trained SOH frozen; new-suite Clean calibration allowed; NOT zero-shot',
        soh_sha256=sha(SOH),upstream_steps='openvla/openvla c8f03f48af692657d3060c19588038c7220e9af9 experiments/robot/libero/run_libero_eval.py')
    path=OUT/'manifest.json';value=dict(protocol=protocol,cases=cases)
    if path.exists(): assert json.loads(path.read_text())==value
    else: write(path,value)
    print('PREPARED',len(cases),protocol['soh_sha256'],flush=True)


def source(shard,workers):
    assert json.loads((OUT/'manifest.json').read_text())['protocol']['soh_sha256']==sha(SOH)
    actor=load_actor();versions=[p._version for p in actor._model.parameters()]
    dest=OUT/'source';dest.mkdir(exist_ok=True)
    suite=Libero10Suite(0);specs=list(suite.iter_episodes());suite.close()
    for task in range(shard,10,workers):
        path=dest/f'task{task}.npz'
        if path.exists() and path.with_suffix('.json').exists(): continue
        case=dict(task_id=task,init_index=0,base_task=specs[task].task)
        env,spec,obs=setup(case);actor.reset(0);seed_torch(0)
        hook=SmolVLAV1Hook(actor._model,layers=LAYERS);means=[];seconds=[];success=False;start=time.monotonic()
        try:
            for step in range(MAX_STEPS):
                hook.begin();action=actor.act(obs);stats=hook.end()
                assert np.isfinite(action).all() and action.shape==(7,)
                for key,bag in [('mean',means),('second',seconds)]:
                    bag.append(torch.stack([torch.stack([stats[l,v][key] for v in range(2)]) for l in LAYERS]).cpu().numpy())
                result=env.step(action);obs=result.obs;success=bool(result.success)
                if step%100==0: print('SOURCE_PROGRESS',task,step,round(time.monotonic()-start,1),flush=True)
                if result.done or success:break
            assert versions==[p._version for p in actor._model.parameters()]
            assert all(not p.requires_grad and p.grad is None for p in actor._model.parameters())
            np.savez_compressed(path,mean=np.stack(means),second=np.stack(seconds))
            write(path.with_suffix('.json'),dict(task_id=task,task=spec.task,init_index=0,success=success,
                steps=len(means),seconds=time.monotonic()-start,moments_sha256=sha(path),max_steps=520))
            print('SOURCE_DONE',task,success,len(means),flush=True)
        finally:hook.close();env.close()


def fit():
    raw={};source_rows=[]
    for t in range(10):
        p=OUT/'source'/f'task{t}.npz';r=json.loads(p.with_suffix('.json').read_text())
        assert r['init_index']==0 and sha(p)==r['moments_sha256']
        with np.load(p) as z:raw[t]={k:torch.from_numpy(z[k].copy()) for k in ('mean','second')}
        assert raw[t]['mean'].shape[1:]==(4,2,768)
        source_rows.append(r)
    refs,audit=fit_loto(raw)
    payload={f'task{t}_layer{l}_{k}':x.numpy() for (t,l),v in refs.items() for k,x in zip(('mu','ss','tau'),v)}
    payload.update(schema=np.array('smolvla_v1_init0_loto_v1'),task_names=np.array([r['task'] for r in source_rows]))
    assert all(np.isfinite(v).all() for k,v in payload.items() if k not in ('schema','task_names'))
    np.savez_compressed(CALIB,**payload)
    diag.SOURCE=OUT;diag.CALIB=CALIB;diag.source_gate(OUT)
    write(OUT/'libero10_source_gate.json',json.loads((OUT/'source_gate.json').read_text()))
    write(OUT/'calibration_audit.json',dict(folds=audit,source=source_rows,calib_sha256=sha(CALIB),
        source_clean_success=sum(r['success'] for r in source_rows),weighting='all actual frames, no success filtering'))
    print('FIT_DONE',sha(CALIB),'SOURCE_CLEAN',sum(r['success'] for r in source_rows),'/10',flush=True)


def configure():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert sha(SOH)==manifest['protocol']['soh_sha256']
    assert sha(CALIB)==json.loads((OUT/'calibration_audit.json').read_text())['calib_sha256']
    shift.setup=setup
    soh=load_soh(SOH)
    soh.requires_grad_(False)
    paired.SmolVLADeepGate=lambda policy,refs,threshold,**kw:SmolVLASOHAdapter(policy,refs,threshold,soh,ema_beta=0.,**kw)
    paired.ARMS=('frozen','soh')
    return manifest,soh,json.loads((OUT/'libero10_source_gate.json').read_text())['tasks']


def preflight():
    m,soh,gates=configure();actor=load_actor();checks=[]
    for group in ('Clean','Sensor'):
        c=next(c for c in m['cases'] if c['group']==group);suite,_,obs=setup(c)
        refs=reference_for(CALIB,c['task_id']);tau=gates[str(c['task_id'])]['threshold']
        funcs=dict(shift.corruption_functions(),glass_blur=paired.glass_blur_exact)
        obs=shift.perturb(obs,c,0,funcs)
        counts=dict(prefill=0,flow=0)
        expert=actor._model.model.vlm_with_expert
        orig_forward=expert.forward;core=actor._model.model;orig_denoise=core.denoise_step
        def forward(*args,**kwargs):
            if kwargs.get('fill_kv_cache'):counts['prefill']+=1
            return orig_forward(*args,**kwargs)
        def denoise(*args,**kwargs):counts['flow']+=1;return orig_denoise(*args,**kwargs)
        try:
            versions=[p._version for p in actor._model.parameters()];sv=[p._version for p in soh.parameters()]
            actor.reset(0);seed_torch(0);f=actor.act(obs)
            for accept in (False,True):
                actor.reset(0);seed_torch(0);counts.update(prefill=0,flow=0)
                expert.forward=forward;core.denoise_step=denoise
                a=SmolVLASOHAdapter(actor._model,refs,tau,soh,ema_beta=0.,decision_override=accept)
                try:a.begin();pred=actor.act(obs);d=a.end()
                finally:a.close();expert.forward=orig_forward;core.denoise_step=orig_denoise
                assert np.isfinite(pred).all() and d['encoder_calls']==(4 if accept else 2)
                assert counts==dict(prefill=1,flow=10),counts
                if not accept:assert np.array_equal(f,pred)
                checks.append(dict(group=group,forced_accept=accept,counts=dict(counts),encoder=d['encoder_calls'],reject_exact=not accept))
            assert versions==[p._version for p in actor._model.parameters()] and sv==[p._version for p in soh.parameters()]
        finally:suite.close()
    write(OUT/'preflight.json',dict(status='PASS',checks=checks,soh_sha256=sha(SOH)))
    print('PREFLIGHT_PASS',flush=True)


def worker(shard,workers):
    assert json.loads((OUT/'preflight.json').read_text())['status']=='PASS'
    m,soh,gates=configure();actor=load_actor();funcs=dict(shift.corruption_functions(),glass_blur=paired.glass_blur_exact)
    dest=OUT/'pairs';dest.mkdir(exist_ok=True)
    for idx,c in enumerate(m['cases']):
        if idx%workers!=shard:continue
        path=dest/f'{c["id"]}.json'
        if path.exists():continue
        refs=reference_for(CALIB,c['task_id']);tau=gates[str(c['task_id'])]['threshold'];rows={};cache={}
        sv=[p._version for p in soh.parameters()]
        for arm in ('frozen','soh'):
            rows[arm]=paired.rollout(actor,c,arm,refs,tau,funcs,cache,max_steps=520)
            print('ARM_DONE',c['id'],arm,rows[arm]['success'],rows[arm]['steps'],flush=True)
        assert sv==[p._version for p in soh.parameters()]
        write(path,dict(seed=0,case=c,arms=rows,pairing=pair_check(rows),soh_sha256=sha(SOH),calib_sha256=sha(CALIB)))
        print('PAIR_DONE',c['id'],flush=True)


def report():
    m=json.loads((OUT/'manifest.json').read_text());rows=[]
    for c in m['cases']:
        row=json.loads((OUT/'pairs'/f'{c["id"]}.json').read_text());assert row['case']==c
        pair_check(row['arms']);assert row['soh_sha256']==m['protocol']['soh_sha256'];rows.append(row)
    paired.ARMS=('frozen','soh')
    groups={g:paired.stats([r for r in rows if r['case']['group']==g]) for g in ('Clean','Sensor')}
    subs={s:paired.stats([r for r in rows if r['case']['subgroup']==s]) for s in ('motion_blur','gaussian_blur','zoom_blur','glass_blur','fog')}
    write(OUT/'summary.json',dict(protocol=m['protocol'],pairs=len(rows),groups=groups,subgroups=subs,
        calibration=json.loads((OUT/'calibration_audit.json').read_text()),preflight=json.loads((OUT/'preflight.json').read_text()),
        raw_sha256={p.name:sha(p) for p in (OUT/'pairs').glob('*.json')}))
    lines=['# SmolVLA SOH v2 libero_10: frozen-weight transfer with Clean calibration','',
        'Spatial SOH weights frozen; libero_10 init0 Clean calibration allowed. Init1 evaluation, seed0. Not zero-shot or independent confirmation.','',
        '| Group | Frozen | SOH | Rescue / harm | Net | Exact p | Task-cluster 95% CI pp |','|---|---:|---:|---:|---:|---:|---|']
    for name,r in list(groups.items())+list(subs.items()):
        lines.append(f"| {name} | {r['frozen']}/{r['n']} | {r['deepgate']}/{r['n']} | {r['rescued']}/{r['harmed']} | {r['net']:+d} | {r['mcnemar_exact_two_sided']:.6g} | {r['task_cluster_bootstrap95_pp']} |")
    lines+=['','Gate details and hashes are in summary.json; all full trajectories are retained in pairs/.',
        'Single-seed exploratory transfer; no Lighting/Background evaluation in this run; no universal geometry guarantee.']
    (OUT/'PHASE4_LIBERO10_TRANSFER.md').write_text('\n'.join(lines)+'\n')
    print('REPORT_DONE',json.dumps(groups),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','source','fit','preflight','worker','report'))
    p.add_argument('--shard',type=int,default=0);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    torch.set_num_threads(1)
    if a.mode=='source':source(a.shard,a.workers)
    elif a.mode=='worker':worker(a.shard,a.workers)
    else:globals()[a.mode]()
