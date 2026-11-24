#!/bin/bash
set -euo pipefail
cd /hy-tmp/patch-diag
export LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/hy-tmp/LIBERO-plus:/hy-tmp/patch-diag:/hy-tmp/patch-diag/repos/CoRP:/hy-tmp/patch-diag/repos/CoRP/scripts
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mode=$1; out=$2; workers=${3:-4}; pids=(); for ((k=0;k<workers;k++)); do /hy-tmp/corp_venv/bin/python -u scripts/invariance_sensitivity_probe_v1.py "$mode" --out "$out" --workers "$workers" --shard "$k" > "/tmp/isp_${mode}_${k}.log" 2>&1 & pids+=("$!"); done
failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done; exit "$failed"
