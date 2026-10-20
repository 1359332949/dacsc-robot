#!/usr/bin/env bash
# 真实 RoboCasa365 atomic61 训练启动器。
# 默认使用新主基线 flow；DDPM 受控基线请显式设置 DACSC_CONFIG_NAME=train_dinov2_bs192。
# 用法：
#   bash scripts/run_real_train.sh
#   DACSC_CONFIG_NAME=train_dinov2_bs192 bash scripts/run_real_train.sh ...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="${DACSC_CONFIG_NAME:-train_dinov2_flow}"
CLIP_PATH="${DACSC_CLIP_PATH:-/hy-tmp/models/clip-vit-large-patch14}"
if [ ! -d "$CLIP_PATH" ]; then
  echo "找不到本地 CLIP: $CLIP_PATH"
  echo "请先运行 scripts/fetch_clip_modelscope.sh（或设置 DACSC_CLIP_PATH）"
  exit 1
fi
export DACSC_CLIP_PATH="$CLIP_PATH"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export OTEL_SDK_DISABLED=true

python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("DACSC: CUDA 不可用，拒绝启动训练（禁止静默 CPU 回退）")
p = torch.cuda.get_device_properties(0)
print(f"DACSC GPU: {torch.cuda.get_device_name(0)} | {p.total_memory/1024**3:.1f} GiB")
PY
cd "$ROOT/repos/diffusion_policy"
python3 -c 'import dacsc.compat; import runpy; runpy.run_path("train.py", run_name="__main__")' \
  --config-name="$CONFIG_NAME" \
  task=robocasa/pretrain_atomic61 \
  "$@"
