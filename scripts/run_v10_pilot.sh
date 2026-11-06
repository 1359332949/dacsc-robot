#!/usr/bin/env bash
# Train one v10 pilot run with a frozen 5-task panel. This is a thin launcher;
# model, data reader, loss and simulator remain the project/official adapters.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
OUT="${OUT:?set OUT=/path/to/run/output}"
STEPS="${STEPS:-30000}"
TASKS="${TASKS:-OpenDrawer OpenCabinet TurnOffStove TurnOnSinkFaucet TurnOnElectricKettle}"
TRAIN_STYLE_MIN="${TRAIN_STYLE_MIN:-11}"
TRAIN_STYLE_MAX="${TRAIN_STYLE_MAX:-50}"
VAL_STYLE_MIN="${VAL_STYLE_MIN:-51}"
VAL_STYLE_MAX="${VAL_STYLE_MAX:-60}"
cd "$ROOT"

exec "$PY" -m dacsc.run_output --out "$OUT" -- \
  "$ROOT/scripts/train_corp_robocasa.py" \
  --tasks $TASKS \
  --steps "$STEPS" \
  --batch-size "${BATCH_SIZE:-96}" \
  --lr "${LR:-5e-5}" \
  --ema-decay "${EMA_DECAY:-0.999}" \
  --task-aux-lambda "${TASK_AUX_LAMBDA:-0.01}" \
  --task-aux-depth "${TASK_AUX_DEPTH:-1}" \
  --seed "${SEED:-42}" \
  --train-augmentation "${TRAIN_AUGMENTATION:-none}" \
  --data-root "${DATA_ROOT:-/hy-tmp/datasets/robocasa365}" \
  --train-style-min "$TRAIN_STYLE_MIN" \
  --train-style-max "$TRAIN_STYLE_MAX" \
  --val-style-min "$VAL_STYLE_MIN" \
  --val-style-max "$VAL_STYLE_MAX" \
  --source-holdout-residue "${SOURCE_HOLDOUT_RESIDUE:-9}" \
  --per-task-eval "${PER_TASK_EVAL:-16}" \
  --eval-every "${EVAL_EVERY:-$STEPS}" \
  --save-every "${SAVE_EVERY:-5000}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --eval-shift "${EVAL_SHIFT:-color_jitter}" \
  --out-dir "$OUT"
