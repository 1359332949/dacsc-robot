#!/usr/bin/env bash
# Fine-tune the frozen V10 baseline with a matched train-time channel method.
# The control, matched-random, and DomainDrop-ViT runs must use identical
# INIT_MODEL, STEPS, BATCH_SIZE, LR, seed, data, and evaluation settings.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
CONFIG_PY="${CONFIG_PY:-python3}"
OUT="${OUT:?set OUT=/path/to/run/output}"
INIT_MODEL="${INIT_MODEL:?set INIT_MODEL=/path/to/baseline/model.pt}"
CHANNEL_METHOD="${CHANNEL_METHOD:?set CHANNEL_METHOD=none|random|domaindrop_vit}"

cd "$ROOT"
eval "$("$CONFIG_PY" -m dacsc.v10_config --shell)"

STEPS="${STEPS:-5000}"
TASKS="${TASKS:-$V10_TASKS}"
BATCH_SIZE="${BATCH_SIZE:-48}"
LR="${LR:-1e-5}"
EMA_DECAY="${EMA_DECAY:-$V10_EMA_DECAY}"
TASK_AUX_LAMBDA="${TASK_AUX_LAMBDA:-$V10_TASK_AUX_LAMBDA}"
TASK_AUX_DEPTH="${TASK_AUX_DEPTH:-$V10_TASK_AUX_DEPTH}"
EVAL_EVERY="${EVAL_EVERY:-$STEPS}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PER_TASK_EVAL="${PER_TASK_EVAL:-$V10_PER_TASK_EVAL}"
TRAIN_STYLE_MIN="${TRAIN_STYLE_MIN:-$V10_TRAIN_STYLE_MIN}"
TRAIN_STYLE_MAX="${TRAIN_STYLE_MAX:-$V10_TRAIN_STYLE_MAX}"
VAL_STYLE_MIN="${VAL_STYLE_MIN:-$V10_VAL_STYLE_MIN}"
VAL_STYLE_MAX="${VAL_STYLE_MAX:-$V10_VAL_STYLE_MAX}"
SOURCE_HOLDOUT_RESIDUE="${SOURCE_HOLDOUT_RESIDUE:-$V10_SOURCE_HOLDOUT_RESIDUE}"
DATA_ROOT="${DATA_ROOT:-/hy-tmp/datasets/robocasa365_oss}"
SEED="${SEED:-$V10_SEED}"
CHANNEL_LAYERS="${CHANNEL_LAYERS:-2 5 8 11}"
CHANNEL_DROP_RATE="${CHANNEL_DROP_RATE:-0.33}"
CHANNEL_LAYER_PROBABILITY="${CHANNEL_LAYER_PROBABILITY:-0.8}"
CHANNEL_CONSISTENCY_WEIGHT="${CHANNEL_CONSISTENCY_WEIGHT:-1.5}"
DOMAIN_LOSS_WEIGHT="${DOMAIN_LOSS_WEIGHT:-1.0}"
DOMAIN_GRL_LAMBDA="${DOMAIN_GRL_LAMBDA:-0.25}"

CACHE_VIDEO="${CACHE_VIDEO:-1}"
if [[ "$CACHE_VIDEO" == "1" ]]; then
  CACHE_VIDEO_TASKS="${CACHE_VIDEO_TASKS-$V10_CACHE_VIDEO_TASKS}"
  if [[ -z "$CACHE_VIDEO_TASKS" ]]; then
    echo "CACHE_VIDEO=1 requires a non-empty CACHE_VIDEO_TASKS" >&2
    exit 2
  fi
  read -r -a CACHE_VIDEO_TASK_ARGS <<< "$CACHE_VIDEO_TASKS"
  CACHE_VIDEO_ARGS=(--cache-video-tasks "${CACHE_VIDEO_TASK_ARGS[@]}")
elif [[ "$CACHE_VIDEO" == "0" ]]; then
  CACHE_VIDEO_ARGS=(--no-cache-video)
else
  echo "CACHE_VIDEO must be 0 or 1" >&2
  exit 2
fi

read -r -a CHANNEL_LAYER_ARGS <<< "$CHANNEL_LAYERS"

exec "$PY" -m dacsc.run_output --out "$OUT" -- \
  "$ROOT/scripts/train_corp_robocasa.py" \
  --tasks $TASKS \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --ema-decay "$EMA_DECAY" \
  --task-aux-lambda "$TASK_AUX_LAMBDA" \
  --task-aux-depth "$TASK_AUX_DEPTH" \
  --seed "$SEED" \
  --train-augmentation none \
  --eval-shift color_jitter \
  "${CACHE_VIDEO_ARGS[@]}" \
  --data-root "$DATA_ROOT" \
  --train-style-min "$TRAIN_STYLE_MIN" \
  --train-style-max "$TRAIN_STYLE_MAX" \
  --val-style-min "$VAL_STYLE_MIN" \
  --val-style-max "$VAL_STYLE_MAX" \
  --source-holdout-residue "$SOURCE_HOLDOUT_RESIDUE" \
  --per-task-eval "$PER_TASK_EVAL" \
  --eval-every "$EVAL_EVERY" \
  --save-every "$SAVE_EVERY" \
  --num-workers "$NUM_WORKERS" \
  --init-model "$INIT_MODEL" \
  --channel-method "$CHANNEL_METHOD" \
  --channel-layers "${CHANNEL_LAYER_ARGS[@]}" \
  --channel-drop-rate "$CHANNEL_DROP_RATE" \
  --channel-layer-probability "$CHANNEL_LAYER_PROBABILITY" \
  --channel-consistency-weight "$CHANNEL_CONSISTENCY_WEIGHT" \
  --domain-loss-weight "$DOMAIN_LOSS_WEIGHT" \
  --domain-grl-lambda "$DOMAIN_GRL_LAMBDA" \
  --out-dir "$OUT"
