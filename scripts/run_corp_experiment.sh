#!/usr/bin/env bash
# One declared budget; sequential seeds/modes and exclusive output ownership.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
OUT="${OUT:?set a new OUT directory}"
STEPS="${STEPS:?declare the total training budget explicitly}"
SEEDS="${SEEDS:-42}"
MODES="${MODES:-none}"
for SEED in $SEEDS; do
  for MODE in $MODES; do
    echo "[seed=$SEED mode=$MODE steps=$STEPS]"
    PYTHONPATH="$ROOT" "$PY" -m dacsc.run_output --out "$OUT/$MODE/seed_$SEED" -- \
      "$ROOT/scripts/train_corp_robocasa.py" --steps "$STEPS" \
      --batch-size 30 --lr 5e-5 --ema-decay "${EMA_DECAY:-0.99}" \
      --num-workers 0 --save-every "${SAVE_EVERY:-1000}" --eval-every "$STEPS" \
      --per-task-eval "${PER_TASK_EVAL:-40}" --eval-batch-size 16 \
      --seed "$SEED" --token-style-mode "$MODE" --eval-shift "${SHIFT:-color_jitter}" \
      --out-dir "$OUT/$MODE/seed_$SEED"
  done
done
echo ALL_DONE
