"""Read-only independent audit of the complete libero_10 transfer evidence."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def main(out,root):
    m=read(out/'manifest.json');s=read(out/'summary.json');gate=read(out/'libero10_source_gate.json')
    calibration=read(out/'calibration_audit.json');preflight=read(out/'preflight.json');lock=read(out/'code_lock.json')
    assert m['protocol']==s['protocol'] and len(m['cases'])==s['pairs']==30
    assert m['protocol']['max_steps']==520 and m['protocol']['ema_beta']==0
    assert m['protocol']['H']==50 and m['protocol']['E']==1 and m['protocol']['Euler']==10
    assert m['protocol']['soh_sha256']==sha(root/'runs/phase4_soh/soh_v2_ckpt.pt')
    assert read(out/'spatial_checkpoint_origin_receipt.json')['soh_sha256']==m['protocol']['soh_sha256']
    assert calibration['calib_sha256']==sha(out/'libero10_init0_calib.npz')
    assert preflight['status']=='PASS' and len(preflight['checks'])==4
    assert preflight['soh_sha256']==m['protocol']['soh_sha256']
    for c in preflight['checks']:
        assert c['counts']==dict(prefill=1,flow=10) and c['encoder']==(4 if c['forced_accept'] else 2)
        if not c['forced_accept']:assert c['reject_exact']
    actual=read(out/'encoder_call_audit.json')
    assert actual['status']=='PASS' and actual['weights_frozen_and_versions_unchanged'] and len(actual['rows'])==6
    assert actual['soh_sha256']==m['protocol']['soh_sha256'] and actual['calib_sha256']==calibration['calib_sha256']
    for row in actual['rows']:
        assert row['actual_calls']==dict(encoder=4 if row['mode']=='forced_accept' else 2,prefill=1,flow=10)
        if row['mode']=='forced_reject':assert row['action_l2_vs_frozen']==0
    for f,h in lock['sha256'].items():assert sha(root/f)==h,('code drift',f)
    with np.load(out/'libero10_init0_calib.npz',allow_pickle=False) as z:
        for t in range(10):
            for l in (3,6,9,12):
                for k in ('mu','ss','tau'):
                    a=z[f'task{t}_layer{l}_{k}'];assert a.shape==(2,768) and np.isfinite(a).all()
    for t in range(10):
        meta=read(out/'source'/f'task{t}.json');p=out/'source'/f'task{t}.npz'
        assert meta['init_index']==0 and meta['max_steps']==520 and meta['moments_sha256']==sha(p)
        with np.load(p,allow_pickle=False) as z:
            assert z['mean'].shape==z['second'].shape==(meta['steps'],4,2,768)
            assert np.isfinite(z['mean']).all() and np.isfinite(z['second']).all()
        assert t not in calibration['folds'][str(t)]['outer']
        assert set(calibration['folds'][str(t)]['outer'])==set(range(10))-{t}
        for hold,fit in calibration['folds'][str(t)]['inner'].items():
            assert set(fit)==set(range(10))-{t,int(hold)}
        g=gate['tasks'][str(t)]
        assert set(g['outer_tasks'])==set(range(10))-{t}
        for hold,fold in g['folds'].items():
            assert set(fold['fit_tasks'])==set(range(10))-{t,int(hold)}
            for inner,fit in fold['channel_threshold_inner7'].items():
                assert set(fit)==set(range(10))-{t,int(hold),int(inner)}
    rows=[];replans=0;equal_inputs=0;roundings=[]
    for c in m['cases']:
        assert c['init_index']==1 and c['group'] in ('Clean','Sensor')
        if c['group']=='Sensor':assert c['severity']==3
        p=out/'pairs'/f'{c["id"]}.json';r=read(p);rows.append(r)
        assert r['case']==c and r['seed']==0 and s['raw_sha256'][p.name]==sha(p)
        assert r['soh_sha256']==m['protocol']['soh_sha256'] and r['calib_sha256']==calibration['calib_sha256']
        f,g=r['arms']['frozen'],r['arms']['soh']
        assert f['start']==g['start'] and f['observed_start']==g['observed_start']
        for arm,a in r['arms'].items():
            assert a['case_id']==c['id'] and a['arm']==arm and a['init_index']==1 and a['policy_seed']==0
            assert a['environment_seed']==c['task_id'] and 0<a['steps']<=520 and len(a['trace'])==a['steps']
            assert a['finite_actions'] and a['weights_frozen_and_versions_unchanged']
            h=hashlib.sha256()
            for step,x in enumerate(a['trace']):
                assert x['step']==step
                act=np.asarray(x['action'],dtype=np.float32);assert act.shape==(7,) and np.isfinite(act).all()
                h.update(act.tobytes());replans+=1
                if arm=='soh':
                    gate_row=x['gate'];assert gate_row['ema_beta']==0 and gate_row['prefix_builds']==1
                    assert gate_row['encoder_calls']==(4 if gate_row['accepted'] else 2)
                    assert gate_row['accepted']==(gate_row['score']>gate_row['threshold'])
                    assert gate_row['threshold']==gate['tasks'][str(c['task_id'])]['threshold']
            assert h.hexdigest()==a['action_sha256']
            for e in a['rounding_events']:
                assert e['pixels']<=64 and e['max_lsb']<=1
                roundings.append(e)
        for a,b in zip(f['trace'],g['trace']):
            assert a['rng_before']==b['rng_before'] and a['rng_after']==b['rng_after']
            if a['snapshot']==b['snapshot'] and a['render_state']==b['render_state'] and not b['gate']['accepted']:
                assert np.array_equal(np.asarray(a['action'],np.float32),np.asarray(b['action'],np.float32))
                equal_inputs+=1
    for name,x in s['groups'].items():
        rr=[r for r in rows if r['case']['group']==name]
        f=np.array([r['arms']['frozen']['success'] for r in rr],bool);g=np.array([r['arms']['soh']['success'] for r in rr],bool)
        res=int((~f&g).sum());harm=int((f&~g).sum())
        assert x['n']==len(rr) and x['frozen']==int(f.sum()) and x['deepgate']==int(g.sum())
        assert x['rescued']==res and x['harmed']==harm and x['net']==res-harm
        p=float(binomtest(res,res+harm,.5).pvalue) if res+harm else 1.
        assert x['mcnemar_exact_two_sided']==p
        total=sum(len(r['arms']['soh']['trace']) for r in rr)
        acc=sum(sum(fr['gate']['accepted'] for fr in r['arms']['soh']['trace']) for r in rr)
        assert x['gate']['total_replans']==total and x['gate']['accepted_replans']==acc
    result=dict(status='PASS',cases=len(rows),replans=replans,same_input_off_action_checks=equal_inputs,
        rounding_events=len(roundings),max_rounding_pixels=max([e['pixels'] for e in roundings],default=0),
        max_rounding_lsb=max([e['max_lsb'] for e in roundings],default=0),
        code_files_verified=len(lock['sha256']),calib_sha256=calibration['calib_sha256'],soh_sha256=m['protocol']['soh_sha256'])
    p=out/'independent_audit.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('/hy-tmp/patch-diag/runs/phase4_libero10_transfer_v1'))
    p.add_argument('--root',type=Path,default=Path('/hy-tmp/patch-diag'));a=p.parse_args();main(a.out,a.root)
