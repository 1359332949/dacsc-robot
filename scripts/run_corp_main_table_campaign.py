"""One bounded Route A campaign: eight workers, Spatial then Object, resume-safe."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    t=p.with_suffix('.pending');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def main(root):
    assert json.loads((root/'preflight.json').read_text())['status']=='PASS'
    assert (root/'latency.json').exists()
    src=root/'source_code';protocol_path=root/'run_protocol.json'
    sources={str(p.relative_to(src)):sha(p) for p in src.rglob('*.py')}
    manifests={s:sha(root/(s+'_manifest.json')) for s in ('spatial','object')}
    cal=json.loads((root/'calibration_receipt.json').read_text())
    key={'source_sha256':sources,'manifest_sha256':manifests,'loto_sha256':cal['loto_sha256'],'global_sha256':cal['global_sha256'],
         'frozen_reference_sha256':cal['frozen_ours_sha256'],'checkpoint_sha256':'f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6',
         'sensor_wrapper_sha256':sha(Path('/hy-tmp/LIBERO-plus/libero/libero/envs/env_wrapper.py')),
         'max_steps':220,'H':12,'E':12,'Euler_steps':10,'workers':8,'seed_indices':[0,1,2],
         'role':'historically accessed benchmark development, not Blind/Held-out','Object_reference_source':'Spatial only'}
    key['fingerprints']={s:hashlib.sha256(json.dumps([s,key],sort_keys=True).encode()).hexdigest() for s in ('spatial','object')}
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==key,'resume protocol/source mismatch'
    else:write(protocol_path,key)
    started=time.monotonic();python='/hy-tmp/corp_venv/bin/python';export=root/'export/docs/data/corp_main_table_v2';export.mkdir(parents=True,exist_ok=True)
    for suite in ('spatial','object'):
        procs=[];logs=[];stage=time.monotonic()
        try:
            for k in range(8):
                log=(root/f'{suite}_worker{k}.log').open('a');logs.append(log)
                procs.append(subprocess.Popen([python,'-u',str(src/'scripts/corp_main_table_5arm.py'),'--out',str(root),
                    '--suite',suite,'--seeds','0,1,2','--workers','8','--shard',str(k)],stdout=log,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in procs):
                failed=[p.returncode for p in procs if p.poll() is not None and p.returncode]
                if failed:raise RuntimeError(f'{suite} worker failure {failed}; retain complete jobs and inspect logs')
                complete=len(list((root/suite/'jobs').glob('*.json')))
                write(root/'status.json',dict(phase='running',suite=suite,jobs_complete=complete,episodes_complete=5*complete,
                    suite_seconds=time.monotonic()-stage,campaign_seconds=time.monotonic()-started,pids=[p.pid for p in procs]));time.sleep(5)
            assert all(p.returncode==0 for p in procs)
        finally:
            for p in procs:
                if p.poll() is None:p.terminate()
            for p in procs:p.wait()
            for log in logs:log.close()
        subprocess.run([python,str(src/'scripts/analyze_corp_main_table.py'),'--out',str(root),'--suites',
            'spatial' if suite=='spatial' else 'spatial,object','--export',str(export)],check=True)
        write(root/(suite+'_complete.json'),dict(status='complete',jobs=300,episodes=1500,seconds=time.monotonic()-stage))
        print('SUITE_COMPLETE',suite,flush=True)
    write(root/'status.json',dict(phase='complete',suites=['spatial','object'],jobs=600,episodes=3000,seconds=time.monotonic()-started))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    try:main(a.out)
    except Exception as e:write(a.out/'status.json',dict(phase='failed',error=str(e)));raise
