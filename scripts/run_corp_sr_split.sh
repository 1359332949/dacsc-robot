#!/usr/bin/env bash
# source/val/test success-rate for one model, one shift.
set -euo pipefail
OUT="${OUT:?set OUT=/path/to/out_root}"
export MODEL="${MODEL:?set MODEL=/path/to/model.pt}"
export EPISODES="${EPISODES:-20}"
export TASKS="${TASKS:-OpenDrawer OpenCabinet TurnOffStove TurnOnSinkFaucet TurnOnElectricKettle}"
export SHIFT="${SHIFT:-none}"
export MANIFEST_ROOT="${MANIFEST_ROOT:?set MANIFEST_ROOT=/path/to/frozen/eval_manifest}"
SOURCE_STYLES="${SOURCE_STYLES:-11-50}"
VAL_STYLES="${VAL_STYLES:-51-60}"
TARGET_STYLES="${TARGET_STYLES:-1-10}"
# Pilot/model selection uses source/val; test requires an explicit final evaluation.
read -r -a EVAL_SPLITS <<< "${SPLITS:-source val}"
read -r -a EVAL_TASKS <<< "$TASKS"
for split in "${EVAL_SPLITS[@]}"; do
  case "$split" in
    source) styles="$SOURCE_STYLES"; scene_split=pretrain ;;
    val) styles="$VAL_STYLES"; scene_split=pretrain ;;
    target) styles="$TARGET_STYLES"; scene_split=target ;;
    *) echo "Unknown split: $split" >&2; exit 1 ;;
  esac
  STYLES="$styles" SCENE_SPLIT="$scene_split" OUT="$OUT/$split" \
    MANIFEST_DIR="$MANIFEST_ROOT/$split" \
    bash "$(dirname "$0")/run_corp_sr_tasks.sh"
done
"${PY:-python3}" "$(dirname "$0")/aggregate_sr.py" "$OUT" --tasks "${EVAL_TASKS[@]}" \
  --episodes "$EPISODES" --splits "${EVAL_SPLITS[@]}" \
  --style-ranges "source=$SOURCE_STYLES" "val=$VAL_STYLES" \
  "target=$TARGET_STYLES"
echo SPLIT_DONE
