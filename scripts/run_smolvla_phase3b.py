"""One authorized bounded 100-episode run; durable through SSH disconnects."""
import argparse,json,os,subprocess,time
from pathlib import Path


def write(path,value):
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main(a):
    assert json.loads((a.out/'preflight.json').read_text())['status']=='PASS'
    assert len(json.loads((a.out/'manifest.json').read_text())['cases'])==50
    assert not (a.out/'status.json').exists()
    python='/hy-tmp/smolvla_venv/bin/python';src=a.out/'source_code/scripts';started=time.monotonic()
    processes=[];logs=[]
    try:
        for shard in range(4):
            log=(a.out/f'worker{shard}.log').open('w');logs.append(log)
            processes.append(subprocess.Popen([python,'-u',str(src/'phase3_smolvla_shift.py'),'worker','--out',str(a.out),
                '--shard',str(shard),'--workers','4'],stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            fail=[(p.pid,p.returncode) for p in processes if p.poll() is not None and p.returncode]
            if fail:raise RuntimeError(f'worker failure {fail}; results retained, no automatic retry')
            write(a.out/'status.json',{'phase':'running','seconds':time.monotonic()-started,
                'arms_complete':len(list((a.out/'pairs').glob('*_frozen.json')))+len(list((a.out/'pairs').glob('*_v1.json'))),
                'pairs_verified':len(list((a.out/'pairs').glob('*_pair.json'))),
                'pids':[p.pid for p in processes]})
            time.sleep(5)
        assert all(p.returncode==0 for p in processes)
        assert len(list((a.out/'pairs').glob('*_pair.json')))==50
        subprocess.run([python,str(src/'report_smolvla_phase3b.py'),'--run',str(a.out),
            '--phase3a','/hy-tmp/patch-diag/runs/phase3_smolvla','--json',str(a.out/'summary.json'),
            '--md',str(a.out/'PHASE3_SMOLVLA_SHIFT_PILOT.md')],check=True)
        write(a.out/'status.json',{'phase':'complete','seconds':time.monotonic()-started,'episodes':100,'verified_pairs':50})
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait()
        for f in logs:f.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    try:main(a)
    except Exception as e:
        write(a.out/'failure.json',{'error':str(e)});write(a.out/'status.json',{'phase':'failed','error':str(e)});raise
