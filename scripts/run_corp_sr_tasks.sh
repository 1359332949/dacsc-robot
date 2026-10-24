#!/usr/bin/env bash
# Multi-task success-rate evaluation for CoRP-RoboCasa.
set -uo pipefail
ROOT="/root/dacsc-robot"
MODEL="${MODEL:?set MODEL=/path/to/model.pt}"
OUT="${OUT:?set OUT=/path/out}"
TASKS="${TASKS:?set TASKS='CloseFridge OpenDrawer ...'}"
STYLES="${STYLES:-11-30}"
EPISODES="${EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-600}"
mkdir -p "$OUT"
for T in $TASKS; do
  MUJOCO_GL=egl python3 "$ROOT/scripts/eval_corp_robocasa_sr.py" \
    --model "$MODEL" --task "$T" --styles "$STYLES" --episodes "$EPISODES" \
    --max-steps "$MAX_STEPS" --out "$OUT/$T.json" > "$OUT/$T.log" 2>&1
done
echo ALL_DONE
