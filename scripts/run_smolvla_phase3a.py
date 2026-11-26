"""One bounded Phase3A campaign, four GPU processes; never launches the main table."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path


def write(path,data):
    p=path.with_suffix('.pending');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(path)


def main(a):
    root=a.out;src=root/'source_code';python='/hy-tmp/smolvla_venv/bin/python'
    pre=json.loads((root/'preflight.json').read_text());assert pre['status']=='PASS'
    assert 'Ran 6 tests' in (root/'unit_tests_final.log').read_text() and '\nOK' in (root/'unit_tests_final.log').read_text()
    assert not (root/'protocol.json').exists()
    weights=Path('/hy-tmp/upstream_repro/hf-cache/models--HuggingFaceVLA--smolvla_libero/snapshots/6721902bc4d61e50a3bfdb11dfb4cb626f05d102/model.safetensors')
    h=hashlib.sha256()
    with weights.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    assert h.hexdigest()=='71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f'
    protocol={'candidate':'SigLIP layer3, alpha.75, bounded scale[.5,2], nested95%LOTO, noEMA',
        'source':{'tasks':list(range(10)),'init':0,'success_filter':False,'layers':[3,6,9,12]},
        'guard':{'tasks':list(range(10)),'init':1,'arms':['frozen','v1'],'policy_seed':0,
                 'role':'previously examined development Clean, calibration-state-disjoint, NOT untouched confirmation'},
        'scan':{'layers':[3,6,9,12],'alpha':.75,'scope':'cached t0 Clean action drift and identity, not physical geometry/shift success'},
        'rng':'external Torch CPU/CUDA reset after upstream environment reset; common-prefix RNG hashes asserted',
        'model':{'revision':'6721902bc4d61e50a3bfdb11dfb4cb626f05d102','load_vlm_weights':True,
                 'vision_dtype':'BF16','operator_dtype':'FP32 then BF16','n_action_steps':1,'chunk_size':50,'Euler_steps':10},
        'environment':{'suite':'vanilla libero_spatial','renderer':'EGL','noise':0,'max_steps':220,'settling':10},
        'latency':{'rounds':5,'interleaved':True,'gpu_isolated_after_rollouts':True},
        'main_table_autostart':False,'blind_or_heldout_manifests_accessed':False,
        'source_sha256':{str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*.py')}}
    write(root/'protocol.json',protocol)
    began=time.monotonic();durations={}
    def stage(name,script,mode,parallel=False):
        write(root/'status.json',{'phase':name,'elapsed_seconds':time.monotonic()-began})
        t=time.monotonic();processes=[];logs=[]
        try:
            for shard in range(4 if parallel else 1):
                cmd=[python,'-u',str(src/script),mode,'--out',str(root)]
                if parallel:cmd+=['--workers','4','--shard',str(shard)]
                log=(root/f'{name}_{shard}.log').open('w');logs.append(log)
                processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in processes):
                failed=[p.returncode for p in processes if p.poll() is not None and p.returncode]
                if failed:raise RuntimeError(f'{name} worker exit {failed}; inspect retained logs')
                time.sleep(2)
            assert all(p.returncode==0 for p in processes),[(p.pid,p.returncode) for p in processes]
        finally:
            for p in processes:
                if p.poll() is None:p.terminate()
            for p in processes:p.wait()
            for f in logs:f.close()
        durations[name]=time.monotonic()-t
        write(root/'durations.json',durations)
    stage('calibration','phase3_build_smolvla_calib.py','source',True)
    stage('fit','phase3_build_smolvla_calib.py','fit')
    stage('guard','phase3_smolvla_pilot.py','guard',True)
    stage('latency','phase3_smolvla_pilot.py','latency')
    write(root/'status.json',{'phase':'complete','elapsed_seconds':time.monotonic()-began,'durations':durations})
    print('PHASE3A_COMPLETE',json.dumps(durations),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except Exception as e:
        write(a.out/'error.json',{'error':str(e)});write(a.out/'status.json',{'phase':'failed','error':str(e)});raise
