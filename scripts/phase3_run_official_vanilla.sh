#!/bin/bash
set -euo pipefail
cd /hy-tmp/patch-diag
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH=/root/.libero
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH=/hy-tmp/phase3_patch:/hy-tmp/smolvla_venv/lib/python3.10/site-packages:/hy-tmp/corp_venv/lib/python3.10/site-packages:/hy-tmp/LIBERO-plus:/hy-tmp/patch-diag:/hy-tmp/patch-diag/repos/CoRP/scripts:/hy-tmp/patch-diag/repos/CoRP
/hy-tmp/smolvla_venv/bin/lerobot-eval \
 --policy.path=/hy-tmp/smolvla_ckpt/HuggingFaceVLA_smolvla_libero \
 --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
 --env.init_states=true --env.control_mode=relative \
 --env.observation_width=256 --env.observation_height=256 \
 --eval.n_episodes=1 --eval.batch_size=1 --seed=0 \
 --output_dir=/hy-tmp/runs/phase3/official_vanilla > /tmp/official_vanilla.log 2>&1
echo EXIT:$? >> /tmp/official_vanilla.log
