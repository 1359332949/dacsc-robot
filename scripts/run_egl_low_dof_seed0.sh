#!/bin/bash
set -euo pipefail
cd /hy-tmp/patch-diag
export LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/hy-tmp/LIBERO-plus:/hy-tmp/patch-diag:/hy-tmp/patch-diag/repos/CoRP:/hy-tmp/patch-diag/repos/CoRP/scripts
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=runs/egl_low_dof_04
py=/hy-tmp/corp_venv/bin/python
$py scripts/egl_low_dof.py stats --out "$out" > /tmp/ld4_stats.log 2>&1
ps=(); for k in 0 1 2 3; do $py -u scripts/egl_low_dof.py pilot --out "$out" --workers 4 --shard "$k" > /tmp/ld4_pilot_${k}.log 2>&1 & ps+=("$!"); done
fail=0; for p in "${ps[@]}"; do wait "$p" || fail=1; done; test "$fail" = 0
$py scripts/check_low_dof_pilot.py "$out"
ps=(); for k in $(seq 0 7); do $py -u scripts/egl_low_dof.py run --out "$out" --workers 8 --shard "$k" > /tmp/ld4_run_${k}.log 2>&1 & ps+=("$!"); done
fail=0; for p in "${ps[@]}"; do wait "$p" || fail=1; done; exit "$fail"
