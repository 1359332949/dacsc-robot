#!/bin/bash
# Reuse completed groups only after a stopped server job; leave all raw trials intact.
set -euo pipefail
cd /hy-tmp/patch-diag
out=runs/full_traj_v1
exec 9>"$out/resume.lock"; flock -n 9 || exit 2
if pgrep -f '^/hy-tmp/corp_venv/bin/python -u scripts/egl_block3_full_traj.py ' >/dev/null || pgrep -f '^bash scripts/run_egl_full_traj.sh ' >/dev/null; then echo 'original job is active; do not resume'; exit 2; fi
export LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/hy-tmp/LIBERO-plus:/hy-tmp/patch-diag:/hy-tmp/patch-diag/repos/CoRP:/hy-tmp/patch-diag/repos/CoRP/scripts
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
stamp=$(date +%Y%m%d_%H%M%S); preserve="$out/resume_$stamp"; mkdir "$preserve"
python - "$out" "$preserve" <<'PY'
import hashlib,json,pathlib,sys
p,save=map(pathlib.Path,sys.argv[1:]);ref=hashlib.sha256((p/'full.npz').read_bytes()).hexdigest();assert ref==(p/'full.sha256').read_text()=='f5aa659cd4b4fe7e02a512bb1f884ebf19a6711c87f3c62831ddca004b11b218'
m=json.loads(pathlib.Path('docs/data/static_visual_v2_suite/final/dev_manifest.json').read_text());assert m['kind']=='dev';expected={(reg,s,c['case_id']):c for c in m['cases'] for s in range(3) for reg in (['standard','dual'] if c['factor']=='sensor_noise' else ['standard'])};assert len(expected)==390
files=list((p/'trials').glob('*.json'));have=set()
for f in files:
    r=json.loads(f.read_text());key=(r['regime'],r['seed'],r['case_id']);c=expected[key];assert f.name=='%s_%s_%s.json'%key and key not in have and r['full_reference']==ref;have.add(key);names=['frozen','dead075','full_traj_dead075']+(['early_0_2','late_3_plus'] if c['factor']=='background' and c['sub_dimension']=='table_texture' else []);assert [v['method'] for v in r['arms']]==names
for claim in (p/'claims').iterdir():
    if not (p/'trials'/(claim.name+'.json')).exists():claim.rename(save/claim.name)
if (p/'status').exists():(p/'status').rename(save/'previous_status')
print('verified and retained',len(have),'complete groups;',390-len(have),'remaining')
PY
pids=();for ((k=0;k<16;k++));do /hy-tmp/corp_venv/bin/python -u scripts/egl_block3_full_traj.py run --out "$out" --workers 16 --shard "$k" >"$preserve/worker_$k.log" 2>&1 & pids+=("$!");done
fail=0;for pid in "${pids[@]}";do wait "$pid" || fail=1;done;echo "$fail" > "$out/status";exit "$fail"
