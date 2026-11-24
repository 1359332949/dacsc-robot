#!/bin/bash
set -euo pipefail
cd /hy-tmp/patch-diag
export LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/hy-tmp/LIBERO-plus:/hy-tmp/patch-diag:/hy-tmp/patch-diag/repos/CoRP:/hy-tmp/patch-diag/repos/CoRP/scripts
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
out=${1:?new directory}; mkdir "$out"; mkdir "$out/logs"; py=/hy-tmp/corp_venv/bin/python
pool() { mode=$1; workers=$2; pids=(); for ((k=0;k<workers;k++));do "$py" -u scripts/egl_block3_full_traj.py "$mode" --out "$out" --shard "$k" --workers "$workers" > "$out/logs/${mode}_${k}" 2>&1 & pids+=("$!");done; fail=0;for pid in "${pids[@]}";do wait "$pid" || fail=1;done;return "$fail"; }
trap 'echo $? > "$out/status"' EXIT
pool source 5
"$py" -u scripts/egl_block3_full_traj.py fit --out "$out" > "$out/logs/fit" 2>&1
pool run 16
