"""One predeclared severity3 dose diagnostic, same 20 severity5 conditions.

This is development-only, one policy seed; no automatic method or suite expansion.
"""
import argparse,ast,hashlib,json,math,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
import phase3_smolvla_shift as base
from phase3_smolvla_common import predict_at
from dacsc.plus_glass_exact import glass_blur_exact
from report_smolvla_phase3b import summarize

OLD=Path('/hy-tmp/patch-diag/runs/phase3_smolvla_shift')
CALIB=Path('/hy-tmp/patch-diag/runs/phase3_smolvla/spatial_init0_calib.npz')
PYTHON='/hy-tmp/smolvla_venv/bin/python'


def write(path,value):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def validate(out):
    m=json.loads((out/'manifest.json').read_text())
    assert base.sha(CALIB)==m['calib_sha256']
    for name,digest in m['source_sha256'].items():assert base.sha(out/'source_code'/name)==digest,name
    assert base.sha(base.PLUS/'libero/libero/envs/env_wrapper.py')==m['corruption_source_sha256']
    receipt=json.loads((OLD/'glass_exact_receipt.json').read_text())
    assert receipt['status']=='PASS' and len(receipt['checks'])==40
    assert receipt['accelerator_sha256']==base.sha(out/'source_code/dacsc/plus_glass_exact.py')
    return m


def prepare(out):
    old=json.loads((OLD/'manifest.json').read_text());cases=[dict(c,severity=3) for c in old['cases'] if c['group']=='Sensor']
    assert len(cases)==20 and all(c['init_index']==1 and c['policy_seed']==0 for c in cases)
    assert all(sum(c['subgroup']==f for c in cases)==4 for f in base.FAMILIES)
    assert all(sum(c['task_id']==t for c in cases)==2 for t in range(10))
    assert not (out/'manifest.json').exists()
    base.save_json(out/'manifest.json',dict(schema='smolvla_severity3_v1',cases=cases,
        severity5_manifest_sha256=base.sha(OLD/'manifest.json'),calib_sha256=old['calib_sha256'],
        corruption_source_sha256=old['corruption_source_sha256'],
        source_sha256={str(p.relative_to(out/'source_code')):base.sha(p) for p in (out/'source_code').rglob('*.py')},
        protocol=dict(old['protocol'],sensor='official Plus function severity3/10, both cameras from first policy observation',
                      scope='same previously accessed development conditions, new dose; not held-out confirmation',
                      only_scientific_change='sensor function argument 5 -> 3',
                      same_case_ids_for_corruption_seeds=True,policy_seed=0,arms=['frozen','v1'],main_table_autostart=False),
        decision_rule='positive net is an exploratory signal only; nonpositive net closes this fixed SmolVLA candidate; neither outcome proves encoder causation'))


def preflight(out):
    m=validate(out);original=base.corruption_functions();fast=dict(original,glass_blur=glass_blur_exact)
    # Frozen severity5 perturb function is the oracle for parameterization parity.
    path=OLD/'source_code/scripts/phase3_smolvla_shift.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='perturb')
    scope={'np':np,'KEYS':base.KEYS,'isolated_seed':base.isolated_seed,'seed_for':base.seed_for}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<immutable-severity5-perturb>','exec'),scope)
    legacy=scope['perturb'];checks=[];examples={}
    for c in m['cases']:
        suite,_,obs=base.setup(c)
        try:
            oldrow=json.loads((OLD/'pairs'/f"{c['id']}_frozen.json").read_text())
            assert base.snapshot(suite,obs)==oldrow['start'],c['id']
            c5=dict(c,severity=5);before=np.random.get_state()
            old=legacy(obs,c5,0,original);new5=base.perturb(obs,c5,0,fast)
            x=base.perturb(obs,c,0,fast);again=base.perturb(obs,c,0,fast);after=np.random.get_state()
            assert all(np.array_equal(a,b) for a,b in zip(before,after))
            for view,key in enumerate(base.KEYS):
                assert np.array_equal(old[key],new5[key]),(c['id'],key,'severity5 regression')
                assert np.array_equal(x[key],again[key]) and not np.array_equal(x[key],obs[key])
                assert not np.array_equal(x[key],new5[key]),(c['id'],key,'severity not changed')
                from PIL import Image
                with base.isolated_seed(base.seed_for('phase3b',c['id'],0,view)):
                    expected=original[c['subgroup']](Image.fromarray(np.asarray(obs[key])),severity=3).astype(np.uint8)
                assert np.array_equal(x[key],expected),(c['id'],key,'severity3 oracle')
            examples.setdefault(c['subgroup'],(c,x))
            checks.append(dict(case_id=c['id'],raw_start_matches_severity5=True,severity5_parameterization_bit_exact=True,
                               severity3_matches_official_function_both_views=True,both_views_changed=True,
                               severity3_differs_from_severity5=True,outer_numpy_rng_preserved=True))
        finally:suite.close()
    actor=base.load_actor();identity=[]
    for family,(c,obs) in examples.items():
        refs=base.reference_for(CALIB,c['task_id']);f,_=predict_at(actor,obs);z,_=predict_at(actor,obs,refs,3,0,layers=(3,))
        assert np.array_equal(f,z),family;identity.append(family)
    for invalid in (0,11,3.5,True):
        try:base.perturb(obs,dict(c,severity=invalid),0,fast)
        except ValueError:pass
        else:raise AssertionError('invalid severity accepted')
    base.save_json(out/'preflight.json',dict(status='PASS',conditions=checks,native_alpha0_exact_families=identity,
        glass_original_oracle_reused_checks=40,all_twenty_initial_states_and_pixels_match_severity5=True))


def worker(out,shard):
    m=validate(out);assert json.loads((out/'preflight.json').read_text())['status']=='PASS'
    actor=base.load_actor();versions=[p._version for p in actor._model.parameters()]
    funcs=dict(base.corruption_functions(),glass_blur=glass_blur_exact);dest=out/'pairs';dest.mkdir(exist_ok=True)
    for index,c in enumerate(m['cases']):
        if index%4!=shard:continue
        refs=base.reference_for(CALIB,c['task_id']);rows={}
        for arm in (('frozen','v1') if index%2==0 else ('v1','frozen')):
            row=base.rollout(actor,c,arm,refs,funcs);row['sensor_severity']=3
            old=json.loads((OLD/'pairs'/f"{c['id']}_{arm}.json").read_text())
            assert row['start']==old['start'],c['id']
            n=min(row['steps'],old['steps']);assert row['rng_before_replans'][:n]==old['rng_before_replans'][:n]
            base.save_json(dest/f"{c['id']}_{arm}.json",row);rows[arm]=row
            print('ARM_DONE',c['id'],arm,row['success'],row['steps'],round(row['seconds'],1),flush=True)
        f,v=rows['frozen'],rows['v1'];assert f['start']==v['start'] and f['observed_start']==v['observed_start']
        n=min(f['steps'],v['steps']);assert f['rng_before_replans'][:n]==v['rng_before_replans'][:n]
        base.save_json(dest/f"{c['id']}_pair.json",dict(start_exact=True,paired_rng_prefix_exact=True,
                        raw_start_and_flow_rng_match_severity5=True))
    assert versions==[p._version for p in actor._model.parameters()]
    assert all(not p.requires_grad and p.grad is None for p in actor._model.parameters())


def report(out):
    m=validate(out);pairs=[];compact=[]
    for c in m['cases']:
        p=[json.loads((out/'pairs'/f"{c['id']}_{arm}.json").read_text()) for arm in ('frozen','v1')]
        check=json.loads((out/'pairs'/f"{c['id']}_pair.json").read_text());assert all(check.values())
        f,v=p;assert f['start']==v['start'] and f['observed_start']==v['observed_start']
        n=min(f['steps'],v['steps']);assert f['rng_before_replans'][:n]==v['rng_before_replans'][:n]
        for row in p:
            assert row['sensor_severity']==3 and len(row['frames'])==row['steps']==len(row['rng_before_replans'])
            assert all(len(frame['action'])==7 and all(math.isfinite(a) for a in frame['action']) for frame in row['frames'])
        pairs.append(p);compact.append(dict(case_id=c['id'],task_id=c['task_id'],family=c['subgroup'],init_index=1,
            frozen=f['success'],v1=v['success'],steps=[row['steps'] for row in p]))
    assert len(pairs)==20
    groups={g:summarize([p for p in pairs if p[0]['subgroup']==g]) for g in base.FAMILIES};groups['Sensor20']=summarize(pairs)
    g=groups['Sensor20'];decision=('exploratory_positive_net_needs_review' if g['net']>0 else 'close_fixed_SmolVLA_candidate_nonpositive_net')
    result=dict(schema='smolvla_phase3b_severity3_v1',protocol=m['protocol'],groups=groups,cases=compact,
        decision=decision,calib_sha256=m['calib_sha256'],manifest_sha256=base.sha(out/'manifest.json'),
        preflight=json.loads((out/'preflight.json').read_text()),
        paired_checks=dict(pairs=20,initial_physical_and_RGB_exact=True,flow_rng_prefix_exact=True,
                           raw_starts_and_flow_rng_match_severity5=True,finite_actions=True),
        limitations=['one policy seed and four conditions per family; development dose diagnostic, no fresh confirmation',
            'severity3 does not guarantee leaving the floor; zero outcomes do not establish equivalence',
            'nominal exact McNemar ignores task dependence; bootstrap only ten task clusters; zero-difference intervals may degenerate',
            'neither dose outcome identifies SSL-vs-VLM pretraining as a cause; no domain-specific lighting exception',
            'not a SALT matched comparison; original severity5 and Lighting/Clean harms remain reported'])
    base.save_json(out/'summary.json',result)
    lines=['# SmolVLA 双目 Sensor severity=3 补充诊断','',
        '20 条件 × Frozen/V1 = 40 局，固定 V1 Layer3、α=0.75、同一校准包。原任务、init1、策略种子0和逐相机扰动种子均沿用 severity5。',
        '', '| 类别 | Frozen | V1 | 救回 / 误伤 | 净增 | 精确 McNemar p | 任务聚类95%区间(pp) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,x in groups.items():
        lo,hi=x['task_cluster_bootstrap95_pp'];lines.append(f"| {name} | {x['frozen']}/{x['n']} | {x['v1']}/{x['n']} | {x['rescued']} / {x['harmed']} | {x['net']:+d} | {x['mcnemar_exact_two_sided']:.4f} | [{lo:+.1f}, {hi:+.1f}] |")
    lines+=['','| 类别 | Frozen 主/腕激活率 | V1 主/腕激活率 |','|---|---:|---:|']
    for name,x in groups.items():
        f=x['active_fraction_episode_macro']['frozen'];v=x['active_fraction_episode_macro']['v1']
        lines.append(f'| {name} | {100*f[0]:.2f}% / {100*f[1]:.2f}% | {100*v[0]:.2f}% / {100*v[1]:.2f}% |')
    lines+=['',f'判定：`{decision}`。不自动启动大主表、不改参数、不打开封存池。',
            '这是单种子开发诊断；不以此建立“紧凑SSL vs 大规模VLM”的因果边界。原 severity5 全部失败记录与 Lighting/Clean 误伤继续保留。',
            'severity 使用 Plus 噪声函数1–10档参数；数值降为3并不预先保证脱离地板。', '']
    (out/'SMOLVLA_SEVERITY3.md').write_text('\n'.join(lines));print(json.dumps(groups,indent=2),flush=True)


def run(out):
    validate(out);assert json.loads((out/'preflight.json').read_text())['status']=='PASS'
    assert not (out/'status.json').exists();started=time.monotonic();procs=[];logs=[]
    try:
        for shard in range(4):
            log=(out/f'worker{shard}.log').open('w');logs.append(log)
            procs.append(subprocess.Popen([PYTHON,'-u',__file__,'worker','--out',str(out),'--shard',str(shard)],stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in procs):
            failed=[p.returncode for p in procs if p.poll() is not None and p.returncode]
            if failed:raise RuntimeError(f'worker failure {failed}; retain all outputs')
            write(out/'status.json',dict(phase='running',seconds=time.monotonic()-started,
                arms_complete=len(list((out/'pairs').glob('*_frozen.json')))+len(list((out/'pairs').glob('*_v1.json'))),
                verified_pairs=len(list((out/'pairs').glob('*_pair.json'))),pids=[p.pid for p in procs]));time.sleep(5)
        assert all(p.returncode==0 for p in procs)
        report(out);write(out/'status.json',dict(phase='complete',seconds=time.monotonic()-started,episodes=40,verified_pairs=20))
    finally:
        for p in procs:
            if p.poll() is None:p.terminate()
        for p in procs:p.wait()
        for log in logs:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','preflight','worker','run','report'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--shard',type=int,default=0);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    if a.mode=='prepare':prepare(a.out)
    elif a.mode=='preflight':preflight(a.out)
    elif a.mode=='worker':worker(a.out,a.shard)
    elif a.mode=='report':report(a.out)
    else:
        try:run(a.out)
        except Exception as e:write(a.out/'status.json',dict(phase='failed',error=str(e)));raise
