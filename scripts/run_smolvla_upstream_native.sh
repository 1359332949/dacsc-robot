#!/bin/bash
set -euo pipefail
# Executes downloaded, unmodified upstream Python; this file only configures paths.
# Requires the original Hub config and both verified weight files in HF cache.
output=${1:?fresh output JSON path required}
episodes_per_task=${2:-1}
[[ "$episodes_per_task" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 2; }
expected=6e9d9daf1526ff344b3742b935fde167faeffb13
actual=$(git -C /hy-tmp/upstream_repro/roboeval rev-parse HEAD)
[[ "$actual" == "$expected" ]] || { echo "Unexpected upstream revision" >&2; exit 2; }
git -C /hy-tmp/upstream_repro/roboeval diff --exit-code -- src/roboeval scripts/run_smolvla_libero_spatial.py
cmp /hy-tmp/upstream_repro/roboeval/src/roboeval/policies/smolvla.py /hy-tmp/upstream_repro/harness_pkg/roboeval/policies/smolvla.py
mkdir -p "$(dirname "$output")"
export LIBERO_CONFIG_PATH=/hy-tmp/vanilla_libero_config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYGLFW_LIBRARY=/hy-tmp/corp_venv/lib/python3.10/site-packages/glfw/x11/libglfw.so
export HF_HOME=/hy-tmp/upstream_repro/hf-home HF_HUB_CACHE=/hy-tmp/upstream_repro/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=/hy-tmp/upstream_repro/harness_pkg:/hy-tmp/patch-diag/runs/repro_baseline_v2_20261126/author_audit/mujoco327:/hy-tmp/lerobot_community_revision:/hy-tmp/LIBERO-vanilla:/hy-tmp/smolvla_venv/lib/python3.10/site-packages:/hy-tmp/corp_venv/lib/python3.10/site-packages
cd /hy-tmp/upstream_repro/roboeval
exec /hy-tmp/smolvla_venv/bin/python -u scripts/run_smolvla_libero_spatial.py --out "$output" --episodes-per-task "$episodes_per_task" --max-steps 220 --seed 0 --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 --device cuda
