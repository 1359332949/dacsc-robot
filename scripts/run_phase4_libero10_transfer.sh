#!/usr/bin/env bash
set -euo pipefail
source /hy-tmp/libero10_env.sh
cd /hy-tmp/patch-diag
OUT=runs/phase4_libero10_transfer_v1
PY=/hy-tmp/smolvla_venv/bin/python
mkdir -p "$OUT/logs"
trap 'echo FAILED at line $LINENO; date -Iseconds > "$OUT/FAILED"' ERR
"$PY" -u scripts/phase4_libero10_transfer.py prepare
"$PY" - <<'LOCK'
import hashlib,json,shutil
from pathlib import Path
root=Path.cwd();out=root/'runs/phase4_libero10_transfer_v1'
files=['scripts/phase4_libero10_transfer.py','scripts/phase3_smolvla_deepgate_closedloop.py',
       'scripts/phase4_soh_clean5_eval.py','scripts/phase3_smolvla_common.py',
       'scripts/phase3_smolvla_shift.py','scripts/phase3_smolvla_unified_diag.py',
       'dacsc/smolvla_soh.py','dacsc/smolvla_v1_hook.py']
lock={'sha256':{s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in files}}
p=out/'code_lock.json'
if p.exists():assert json.loads(p.read_text())['sha256']==lock['sha256'],'code changed since initial run'
else:p.write_text(json.dumps(lock,indent=2)+'\n')
origin=root/'runs/phase4_soh_v2_3seed/preflight.json'
if not origin.exists():origin=root/'docs/data/phase4/libero10_transfer/spatial_checkpoint_origin_receipt.json'
assert origin.exists(),'Spatial checkpoint origin receipt required'
assert json.loads(origin.read_text())['soh_sha256']==json.loads((out/'manifest.json').read_text())['protocol']['soh_sha256']
shutil.copy2(origin,out/'spatial_checkpoint_origin_receipt.json')
for a,b in [('/hy-tmp/vanilla_libero_config/config.yaml','vanilla_config.yaml'),
            ('/hy-tmp/libero10_env.sh','runtime_env.sh')]:shutil.copy2(a,out/b)
LOCK
pids=()
for k in 0 1 2 3; do
  "$PY" -u scripts/phase4_libero10_transfer.py source --shard "$k" --workers 4 > "$OUT/logs/source_$k.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" -u scripts/phase4_libero10_transfer.py fit > "$OUT/logs/fit.log" 2>&1
"$PY" -u scripts/phase4_libero10_transfer.py preflight > "$OUT/logs/preflight.log" 2>&1
"$PY" -u scripts/audit_phase4_libero10_encoder.py > "$OUT/logs/encoder_audit.log" 2>&1
pids=()
for k in 0 1 2 3; do
  "$PY" -u scripts/phase4_libero10_transfer.py worker --shard "$k" --workers 4 > "$OUT/logs/worker_$k.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" -u scripts/phase4_libero10_transfer.py report > "$OUT/logs/report.log" 2>&1
"$PY" -u scripts/audit_phase4_libero10_transfer.py > "$OUT/logs/independent_audit.log" 2>&1
date -Iseconds > "$OUT/COMPLETE"
echo COMPLETE
