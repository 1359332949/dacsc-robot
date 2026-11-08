#!/usr/bin/env bash
# Train one v10 pilot run with a frozen 5-task panel. This is a thin launcher;
# model, data reader, loss and simulator remain the project/official adapters.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
CONFIG_PY="${CONFIG_PY:-python3}"
OUT="${OUT:?set OUT=/path/to/run/output}"
cd "$ROOT"
eval "$("$CONFIG_PY" -m dacsc.v10_config --shell)"

STEPS="${STEPS:-$V10_STEPS}"
TASKS="${TASKS:-$V10_TASKS}"
TRAIN_STYLE_MIN="${TRAIN_STYLE_MIN:-$V10_TRAIN_STYLE_MIN}"
TRAIN_STYLE_MAX="${TRAIN_STYLE_MAX:-$V10_TRAIN_STYLE_MAX}"
VAL_STYLE_MIN="${VAL_STYLE_MIN:-$V10_VAL_STYLE_MIN}"
VAL_STYLE_MAX="${VAL_STYLE_MAX:-$V10_VAL_STYLE_MAX}"
# CACHE_VIDEO=0 streams through the official DP reader. The default keeps the
# official RoboCasa cache only for named tasks below the GPUSHARE cgroup cap.
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

exec "$PY" -m dacsc.run_output --out "$OUT" -- \
  "$ROOT/scripts/train_corp_robocasa.py" \
  --tasks $TASKS \
  --steps "$STEPS" \
  --batch-size "${BATCH_SIZE:-$V10_BATCH_SIZE}" \
  --lr "${LR:-$V10_LR}" \
  --ema-decay "${EMA_DECAY:-$V10_EMA_DECAY}" \
  --task-aux-lambda "${TASK_AUX_LAMBDA:-$V10_TASK_AUX_LAMBDA}" \
  --task-aux-depth "${TASK_AUX_DEPTH:-$V10_TASK_AUX_DEPTH}" \
  --seed "${SEED:-$V10_SEED}" \
  --train-augmentation "${TRAIN_AUGMENTATION:-none}" \
  "${CACHE_VIDEO_ARGS[@]}" \
  --data-root "${DATA_ROOT:-/hy-tmp/datasets/robocasa365}" \
  --train-style-min "$TRAIN_STYLE_MIN" \
  --train-style-max "$TRAIN_STYLE_MAX" \
  --val-style-min "$VAL_STYLE_MIN" \
  --val-style-max "$VAL_STYLE_MAX" \
  --source-holdout-residue "${SOURCE_HOLDOUT_RESIDUE:-$V10_SOURCE_HOLDOUT_RESIDUE}" \
  --per-task-eval "${PER_TASK_EVAL:-$V10_PER_TASK_EVAL}" \
  --eval-every "${EVAL_EVERY:-$STEPS}" \
  --save-every "${SAVE_EVERY:-$V10_SAVE_EVERY}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --eval-shift "${EVAL_SHIFT:-color_jitter}" \
  --out-dir "$OUT"
