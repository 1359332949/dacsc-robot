#!/usr/bin/env bash
# 真实 RoboCasa365 atomic61 训练启动器。
# 用法：
#   bash scripts/run_real_train.sh
#   bash scripts/run_real_train.sh dataloader.batch_size=32 training.max_train_steps=20000
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="${DACSC_CONFIG_NAME:-train_dinov2_bs192}"
CLIP_PATH="${DACSC_CLIP_PATH:-/hy-tmp/models/clip-vit-large-patch14}"
if [ ! -d "$CLIP_PATH" ]; then
  echo "找不到本地 CLIP: $CLIP_PATH"
  echo "请先运行 scripts/fetch_clip_modelscope.sh（或设置 DACSC_CLIP_PATH）"
  exit 1
fi
export DACSC_CLIP_PATH="$CLIP_PATH"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_SILENT=true
cd "$ROOT/repos/diffusion_policy"
python3 -c 'import dacsc.compat; import runpy; runpy.run_path("train.py", run_name="__main__")' \
  --config-name="$CONFIG_NAME" \
  task=robocasa/pretrain_atomic61 \
  "$@"
