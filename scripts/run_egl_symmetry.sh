#!/bin/bash
set -euo pipefail
cd /hy-tmp/patch-diag
export LIBERO_CONFIG_PATH=/root/.libero PYTHONPATH=/hy-tmp/LIBERO-plus:/hy-tmp/patch-diag:/hy-tmp/patch-diag/repos/CoRP:/hy-tmp/patch-diag/repos/CoRP/scripts
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
py=/hy-tmp/corp_venv/bin/python; out=runs/egl_symmetry_v1
mkdir "$out"
for mode in standard wrist; do $py -u scripts/egl_trust_symmetry.py "$mode" --out "${out}_pilot" --limit 1 > "/tmp/sym_pilot_${mode}.log" 2>&1; done
# Each shard covers the identical planned IDs at each seed. All methods actually run.
pids=()
for k in $(seq 0 7); do
 (for mode in wrist standard; do for seed in 0 1 2; do
 if [ "$mode" = wrist ] && [ "$seed" != 0 ]; then continue; fi
 $py -u scripts/egl_trust_symmetry.py "$mode" --out "$out" --workers 8 --shard "$k" --seed "$seed"
 done; done) > "/tmp/sym_worker_${k}.log" 2>&1 & pids+=("$!")
done
failed=0; for p in "${pids[@]}"; do wait "$p" || failed=1; done; exit "$failed"
