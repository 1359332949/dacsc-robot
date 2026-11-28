"""Check retained Spatial pair counts and actual seeded RNG, plus beta0 metadata.

Read-only: no policy loading, environment rollout, training, or threshold fit.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from scipy.stats import binomtest

ROOT=Path('/hy-tmp/patch-diag')
OUT=ROOT/'runs/phase4_libero10_transfer_v1'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1);torch.cuda.init()
    expected={}
    for s in (0,1,2):
        torch.manual_seed(s);torch.cuda.manual_seed_all(s)
        expected[s]=hashlib.sha256(torch.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
    assert len(set(expected.values()))==3
    directory=ROOT/'runs/phase4_soh_v2_3seed';rows=[];mislabel=0
    cases={};by_seed={s:[] for s in expected}
    for p in sorted((directory/'pairs').glob('*.json')):
        s=int(p.stem.split('_',1)[0][1:]);r=read(p)
        assert s in expected and r.get('seed',s)==s
        assert r['case']['group'] in ('Sensor','Background')
        if r['case']['group']=='Sensor':assert r['case']['severity']==3
        key=r['case']['id'];core={k:r['case'].get(k) for k in ('id','task_id','base_task','init_index','group','subgroup','severity','language')}
        if key in cases:assert cases[key]==core,('condition mismatch',key)
        else:cases[key]=core
        f,g=r['arms']['frozen'],r['arms']['soh']
        assert f['start']==g['start'] and f['observed_start']==g['observed_start']
        for a in (f,g):
            assert a['trace'][0]['rng_before']==expected[s],('seed does not match actual RNG',p,s)
            if a.get('policy_seed')!=s:mislabel+=1
            assert a['finite_actions'] and a['weights_frozen_and_versions_unchanged']
        for a,b in zip(f['trace'],g['trace']):
            assert a['rng_before']==b['rng_before'] and a['rng_after']==b['rng_after']
        rows.append(r);by_seed[s].append(r)
    assert len(rows)==90 and len(cases)==30 and all(len(rr)==30 for rr in by_seed.values())
    sums={};per_seed={}
    for s,rr in [(None,rows),*by_seed.items()]:
        groups={}
        for name in ('Sensor','Background'):
            sub=[r for r in rr if r['case']['group']==name]
            f=np.array([r['arms']['frozen']['success'] for r in sub],bool)
            g=np.array([r['arms']['soh']['success'] for r in sub],bool)
            res=int((~f&g).sum());harm=int((f&~g).sum())
            groups[name]=dict(n=len(sub),frozen=int(f.sum()),soh=int(g.sum()),rescued=res,harmed=harm,net=res-harm,
                p=float(binomtest(res,res+harm,.5).pvalue) if res+harm else 1.)
        if s is None:sums=groups
        else:per_seed[str(s)]=groups
    committed=read(ROOT/'docs/data/phase4/soh_v2_3seed_summary.json')
    for name,x in sums.items():
        for k,v in x.items():assert committed['groups'][name][k]==v,(name,k)
    main_path=ROOT/'runs/phase4_soh/soh_v2_ckpt.pt';beta_path=ROOT/'runs/phase4_soh/soh_v2_beta0_ckpt.pt'
    trained={}
    for name,path,beta in [('beta1',main_path,1.),('beta0',beta_path,0.)]:
        p=torch.load(path,map_location='cpu',weights_only=False);receipt=read(path.with_suffix('.json'))
        assert p['arch']=='v2' and float(p['damping_beta'])==beta
        assert receipt['epochs']==30 and receipt['tasks']==10 and receipt['seed']==0
        trained[name]=dict(sha256=sha(path),damping_beta=beta,epochs=30,tasks=10,seed=0)
    assert read(directory/'preflight.json')['soh_sha256']==trained['beta1']['sha256']
    result=dict(status='PASS',pairs=90,conditions=30,actual_rng_verified_for_seeds=[0,1,2],
        seed_start_rng_sha256={str(k):v for k,v in expected.items()},groups=sums,per_seed=per_seed,
        historical_arm_policy_seed_metadata_mismatches=mislabel,
        metadata_note='Older base.rollout hard-coded arm.policy_seed=0. Outer pair seed/filename and actual Torch CPU+CUDA RNG prove the executed streams. Original evidence retained unchanged.',
        beta_checkpoints=trained,beta_scope='separately trained beta0 checkpoint; retained diagnostic is t0 only, not closed-loop ablation')
    (OUT/'spatial_beta0_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
