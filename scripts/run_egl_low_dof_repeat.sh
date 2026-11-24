#!/bin/bash
# Only seed-0 qualifier: trust. Keep eight total workers on the same GPU.
set -euo pipefail
cd /hy-tmp/patch-diag
export LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/hy-tmp/LIBERO-plus:/hy-tmp/patch-diag:/hy-tmp/patch-diag/repos/CoRP:/hy-tmp/patch-diag/repos/CoRP/scripts
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=runs/egl_low_dof_repeat_01
mkdir "$out"; cp runs/egl_low_dof_04/clean_moments.json "$out"  # not used by trust or Frozen
pids=()
for seed in 1 2; do for k in 0 1 2 3; do
 /hy-tmp/corp_venv/bin/python -u scripts/egl_low_dof.py run --out "$out" --workers 4 --shard "$k" --seed "$seed" --method trust > "/tmp/ld_repeat_${seed}_${k}.log" 2>&1 & pids+=("$!")
done; done
failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done; exit "$failed"
