#!/usr/bin/env bash
# source/val/test success-rate for one model, one shift.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PY="${CONFIG_PY:-python3}"
cd "$ROOT"
eval "$("$CONFIG_PY" -m dacsc.v10_config --shell)"

OUT="${OUT:?set OUT=/path/to/out_root}"
export MODEL="${MODEL:?set MODEL=/path/to/model.pt}"
export EPISODES="${EPISODES:-$V10_EPISODES}"
export TASKS="${TASKS:-$V10_TASKS}"
export SHIFT="${SHIFT:-none}"
export BLUR_SIGMA="${BLUR_SIGMA:-0}"
export MANIFEST_ROOT="${MANIFEST_ROOT:?set MANIFEST_ROOT=/path/to/frozen/eval_manifest}"
SOURCE_STYLES="${SOURCE_STYLES:-$V10_SOURCE_STYLES}"
VAL_STYLES="${VAL_STYLES:-$V10_VAL_STYLES}"
TARGET_STYLES="${TARGET_STYLES:-$V10_TARGET_STYLES}"
export ACTION_STEPS="${ACTION_STEPS:-$V10_ACTION_STEPS}"
export LAYOUT_ID="${LAYOUT_ID:-$V10_LAYOUT_ID}"
DATA_ROOT="${DATA_ROOT:-${DACSC_DATA_ROOT:-/hy-tmp/datasets/robocasa365}}"
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
  if [[ "$scene_split" == "pretrain" && -z "$LAYOUT_ID" ]]; then
    echo "source/val evaluation requires explicit LAYOUT_ID" >&2
    exit 2
  fi
  if [[ "$split" == "source" ]]; then
    mkdir -p "$OUT"
    "$CONFIG_PY" "$(dirname "$0")/check_eval_support.py" \
      --data-root "$DATA_ROOT" \
      --tasks "${EVAL_TASKS[@]}" \
      --styles "$styles" \
      --layout-id "$LAYOUT_ID" \
      --train-style-min "$V10_TRAIN_STYLE_MIN" \
      --train-style-max "$V10_TRAIN_STYLE_MAX" \
      --source-holdout-residue "$V10_SOURCE_HOLDOUT_RESIDUE" |
      tee "$OUT/source_panel_support.json"
  fi
  STYLES="$styles" SCENE_SPLIT="$scene_split" OUT="$OUT/$split" \
    BLUR_SIGMA="$BLUR_SIGMA" \
    MANIFEST_DIR="$MANIFEST_ROOT/$split" \
    bash "$(dirname "$0")/run_corp_sr_tasks.sh"
done
"${PY:-python3}" "$(dirname "$0")/aggregate_sr.py" "$OUT" --tasks "${EVAL_TASKS[@]}" \
  --episodes "$EPISODES" --splits "${EVAL_SPLITS[@]}" \
  --style-ranges "source=$SOURCE_STYLES" "val=$VAL_STYLES" \
  "target=$TARGET_STYLES"
echo SPLIT_DONE
