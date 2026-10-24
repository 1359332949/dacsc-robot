#!/usr/bin/env bash
# 修复后 token_style smoke：四组 mode，同 task/seed/步数。
set -uo pipefail
ROOT="/root/dacsc-robot"
STEPS="${STEPS:-51}"
TASKS="${TASKS:-5}"
OUT="${OUT:-/hy-tmp/runs/token_smoke_51}"
mkdir -p "$OUT"
for MODE in none random_stats cross_style_stats cross_style_stats_action_aware; do
  echo "[mode=$MODE]"
  CORP_REPO=/hy-tmp/models/corp/CoRP PYTHONPATH="$ROOT" python3 "$ROOT/scripts/train_corp_robocasa.py" \
    --steps "$STEPS" --batch-size 8 --eval-batch-size 8 --per-task-eval 8 \
    --max-tasks "$TASKS" --num-workers 0 --eval-every "$STEPS" \
    --token-style-mode "$MODE" --seed 42 \
    --out-dir "$OUT/$MODE" > "$OUT/$MODE.log" 2>&1
done
echo ALL_DONE
