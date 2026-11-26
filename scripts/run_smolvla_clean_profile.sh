#!/bin/bash
set -euo pipefail
cd /hy-tmp/patch-diag
pipeline=${1:?modern or legacy};out=${2:?fresh output};tasks=${3:-0}
export LIBERO_CONFIG_PATH=/hy-tmp/vanilla_libero_config MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYGLFW_LIBRARY=/hy-tmp/corp_venv/lib/python3.10/site-packages/glfw/x11/libglfw.so
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/hy-tmp/LIBERO-vanilla:/hy-tmp/smolvla_venv/lib/python3.10/site-packages:/hy-tmp/corp_venv/lib/python3.10/site-packages
if [[ "$pipeline" = legacy ]];then export PYTHONPATH=/hy-tmp/lerobot_community_revision:$PYTHONPATH;fi
mkdir "$out";exec 9>"$out/lock";flock -n 9
for task in $tasks;do /hy-tmp/smolvla_venv/bin/python -u runs/smolvla_reproduction_v1/smolvla_clean_repro.py --pipeline "$pipeline" --assets vanilla --task "$task" --out "$out" >"$out/task$task.log" 2>&1;done
echo 0 >"$out/status"
