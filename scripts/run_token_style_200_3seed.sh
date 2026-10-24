#!/usr/bin/env bash
# 4 modes x 3 seeds x 200 steps, clean + color_jitter MSE.
set -uo pipefail
ROOT="/root/dacsc-robot"
STEPS="${STEPS:-200}"
TASKS="${TASKS:-0}"
OUT="${OUT:-/hy-tmp/runs/token_style_200_3seed}"
SHIFT="${SHIFT:-color_jitter}"
mkdir -p "$OUT"
for SEED in 42 43 44; do
  (
    for MODE in none random_stats cross_style_stats cross_style_stats_action_aware; do
      D="$OUT/$MODE/seed_$SEED"
      mkdir -p "$D"
      echo "[seed=$SEED mode=$MODE]"
      CORP_REPO=/hy-tmp/models/corp/CoRP PYTHONPATH="$ROOT" python3 "$ROOT/scripts/train_corp_robocasa.py" \
        --steps "$STEPS" --batch-size 16 --eval-batch-size 16 --per-task-eval 16 \
        --max-tasks "$TASKS" --num-workers 0 --eval-every "$STEPS" \
        --token-style-mode "$MODE" --seed "$SEED" --eval-shift "$SHIFT" \
        --out-dir "$D" > "$D/run.log" 2>&1
    done
  ) &
done
wait
echo ALL_DONE
