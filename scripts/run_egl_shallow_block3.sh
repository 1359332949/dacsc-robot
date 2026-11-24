#!/bin/bash
set -euo pipefail
cd /hy-tmp/patch-diag
export LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/hy-tmp/LIBERO-plus:/hy-tmp/patch-diag:/hy-tmp/patch-diag/repos/CoRP:/hy-tmp/patch-diag/repos/CoRP/scripts
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=${1:?fresh run path}; workers=${2:-8}; mkdir "$out"; mkdir "$out/logs"
py=/hy-tmp/corp_venv/bin/python
"$py" -u scripts/egl_shallow_block3_closed_loop.py --pilot --out "$out/pilot" >"$out/logs/pilot" 2>&1
pids=(); for ((k=0;k<workers;k++)); do "$py" -u scripts/egl_shallow_block3_closed_loop.py --out "$out/trials" --shard "$k" --workers "$workers" >"$out/logs/$k" 2>&1 & pids+=("$!"); done
failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done; echo "$failed" > "$out/exit_status"; exit "$failed"
