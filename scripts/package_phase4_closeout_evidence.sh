#!/usr/bin/env bash
set -euo pipefail
source /hy-tmp/libero10_env.sh
cd /hy-tmp/patch-diag
OUT=runs/phase4_libero10_transfer_v1
PY=/hy-tmp/smolvla_venv/bin/python
[[ -f "$OUT/COMPLETE" ]]
"$PY" scripts/audit_phase4_libero10_transfer.py > "$OUT/logs/independent_audit.log" 2>&1
"$PY" scripts/audit_phase4_spatial_beta0.py > "$OUT/logs/spatial_beta0_audit.log" 2>&1
"$PY" -m pip freeze > "$OUT/pip_freeze.txt"
"$PY" - <<'META'
import json,sys,importlib,hashlib
from pathlib import Path
importlib.import_module('lerobot.common.policies.smolvla.modeling_smolvla')
source_dir=Path('runs/phase4_libero10_transfer_v1/upstream_model_sources');source_dir.mkdir(exist_ok=True)
model_sources={}
for name,m in list(sys.modules.items()):
 if name.startswith(('lerobot.common.policies.smolvla','lerobot.policies.smolvla')) and getattr(m,'__file__',None):
  p=Path(m.__file__)
  if p.suffix=='.py':
   data=p.read_bytes();(source_dir/(name.replace('.','_')+'.py')).write_bytes(data)
   model_sources[name]=dict(file=str(p),sha256=hashlib.sha256(data).hexdigest())
mods={}
for name in ['torch','numpy','scipy','mujoco','robosuite','lerobot','roboeval']:
 m=importlib.import_module(name);p=Path(m.__file__) if m.__file__ else None
 mods[name]=dict(version=str(getattr(m,'__version__','unknown')),file=str(p),
                 module_file_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p and p.is_file() else None)
Path('runs/phase4_libero10_transfer_v1/runtime_versions.json').write_text(json.dumps(dict(python=sys.version,modules=mods,model_sources=model_sources),indent=2)+'\n')
META
ARCHIVE=/hy-tmp/phase4_closeout_evidence_20261128.tar.gz
tar -czf "$ARCHIVE" \
  runs/phase4_libero10_transfer_v1 \
  runs/phase4_soh_v2_3seed runs/phase4_soh_beta0_diag \
  runs/phase3_smolvla/source runs/phase3_smolvla/spatial_init0_calib.npz \
  runs/phase3_smolvla_unified_diag_v1/source_gate.json \
  runs/phase4_soh/soh_v2_ckpt.pt runs/phase4_soh/soh_v2_ckpt.json \
  runs/phase4_soh/soh_v2_beta0_ckpt.pt runs/phase4_soh/soh_v2_beta0_ckpt.json \
  runs/phase4_soh/subspace_7dof_init0.npz runs/phase4_soh/subspace_7dof_init0.json \
  scripts/phase4_libero10_transfer.py scripts/phase3_smolvla_deepgate_closedloop.py \
  scripts/phase4_soh_clean5_eval.py scripts/phase3_smolvla_common.py \
  scripts/phase3_smolvla_shift.py scripts/phase3_smolvla_unified_diag.py \
  dacsc/smolvla_soh.py dacsc/smolvla_v1_hook.py \
  scripts/audit_phase4_libero10_transfer.py scripts/audit_phase4_libero10_encoder.py \
  scripts/audit_phase4_spatial_beta0.py scripts/phase4_soh_train.py \
  -C /hy-tmp/upstream_repro/harness_pkg roboeval
sha256sum "$ARCHIVE"
stat -c 'archive_bytes=%s' "$ARCHIVE"
