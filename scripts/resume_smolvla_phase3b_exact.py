"""Resume completed arms unchanged; exact CPU glass acceleration, same pilot.

Model, observations, corruption pixels, flow RNG, cases and references unchanged.
Work stealing is scheduling only; it never selects cases using their outcomes.
"""
import argparse,json,os,subprocess,time
from pathlib import Path
from phase3_smolvla_shift import (load_actor,reference_for,corruption_functions,rollout,sha,save_json)
from dacsc.plus_glass_exact import glass_blur_exact


def write(path,value):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def worker(out):
    m=json.loads((out/'manifest.json').read_text())
    receipt=json.loads((out/'glass_exact_receipt.json').read_text())
    assert receipt['status']=='PASS' and len(receipt['checks'])==40
    assert receipt['accelerator_sha256']==sha(out/'source_code/dacsc/plus_glass_exact.py')
    for path,digest in m['source_sha256'].items():assert sha(out/'source_code'/path)==digest,path
    assert m['calib_sha256']==sha('/hy-tmp/patch-diag/runs/phase3_smolvla/spatial_init0_calib.npz')
    actor=load_actor();versions=[p._version for p in actor._model.parameters()]
    functions=corruption_functions();functions['glass_blur']=glass_blur_exact
    # Start uncompleted expensive glass cases first. The claim queue only uses
    # completion files and family names, never success/failure values.
    plan=sorted(enumerate(m['cases']),key=lambda x:(x[1]['subgroup']!='glass_blur',x[0]))
    dest=out/'pairs';claims=out/'resume_claims';claims.mkdir(exist_ok=True)
    for index,case in plan:
        if (dest/f"{case['id']}_pair.json").exists():continue
        claim=claims/case['id']
        try:fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError:continue
        with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
        refs=reference_for('/hy-tmp/patch-diag/runs/phase3_smolvla/spatial_init0_calib.npz',case['task_id'])
        rows={}
        for arm in (('frozen','v1') if index%2==0 else ('v1','frozen')):
            path=dest/f"{case['id']}_{arm}.json"
            if path.exists():rows[arm]=json.loads(path.read_text());continue
            row=rollout(actor,case,arm,refs,functions)
            row['cpu_sensor_backend']='bit-exact accelerated glass; other official functions unchanged'
            save_json(path,row);rows[arm]=row
            print('ARM_DONE',case['id'],arm,row['success'],row['steps'],round(row['seconds'],1),flush=True)
        r0,r1=rows['frozen'],rows['v1'];assert r0['start']==r1['start'] and r0['observed_start']==r1['observed_start'],case['id']
        n=min(r0['steps'],r1['steps']);assert r0['rng_before_replans'][:n]==r1['rng_before_replans'][:n],case['id']
        save_json(dest/f"{case['id']}_pair.json",dict(case_id=case['id'],start_exact=True,paired_rng_prefix_exact=True))
    assert versions==[p._version for p in actor._model.parameters()]
    assert all(not p.requires_grad and p.grad is None for p in actor._model.parameters())


def campaign(out):
    receipt=json.loads((out/'glass_exact_receipt.json').read_text());assert receipt['status']=='PASS'
    assert json.loads((out/'status.json').read_text())['phase']=='stopped_for_exact_cpu_acceleration'
    started=time.monotonic();python='/hy-tmp/smolvla_venv/bin/python';processes=[];logs=[]
    try:
        for shard in range(4):
            log=(out/f'resume_worker{shard}.log').open('w');logs.append(log)
            processes.append(subprocess.Popen([python,'-u',__file__,'worker','--out',str(out)],stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            failed=[(p.pid,p.returncode) for p in processes if p.poll() is not None and p.returncode]
            if failed:raise RuntimeError(f'worker failure {failed}; no automatic retries')
            write(out/'status.json',dict(phase='running_exact_cpu_resume',resume_seconds=time.monotonic()-started,
                arms_complete=len(list((out/'pairs').glob('*_frozen.json')))+len(list((out/'pairs').glob('*_v1.json'))),
                pairs_verified=len(list((out/'pairs').glob('*_pair.json'))),pids=[p.pid for p in processes]))
            time.sleep(5)
        assert all(p.returncode==0 for p in processes)
        assert len(list((out/'pairs').glob('*_pair.json')))==50
        subprocess.run([python,str(out/'source_code/scripts/report_smolvla_phase3b.py'),'--run',str(out),
            '--phase3a','/hy-tmp/patch-diag/runs/phase3_smolvla','--json',str(out/'summary.json'),
            '--md',str(out/'PHASE3_SMOLVLA_SHIFT_PILOT.md')],check=True)
        write(out/'status.json',dict(phase='complete',resume_seconds=time.monotonic()-started,episodes=100,verified_pairs=50))
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait()
        for log in logs:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('worker','campaign'));p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    if a.mode=='worker':worker(a.out)
    else:
        try:campaign(a.out)
        except Exception as e:
            write(a.out/'resume_failure.json',{'error':str(e)});write(a.out/'status.json',{'phase':'failed','error':str(e)});raise
