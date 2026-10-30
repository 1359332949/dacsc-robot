#!/usr/bin/env bash
# source/val/test success-rate for one model, one shift.
set -euo pipefail
OUT="${OUT:?set OUT=/path/to/out_root}"
export MODEL="${MODEL:?set MODEL=/path/to/model.pt}"
export EPISODES="${EPISODES:-10}"
export TASKS="${TASKS:-CloseFridge OpenFridge CloseDrawer OpenDrawer CloseCabinet OpenCabinet TurnOnStove TurnOffStove PickPlaceCounterToCabinet PickPlaceCabinetToCounter}"
export SHIFT="${SHIFT:-none}"
export MANIFEST_ROOT="${MANIFEST_ROOT:?set MANIFEST_ROOT=/path/to/frozen/eval_manifest}"
# Pilot/model selection uses source/val; test requires an explicit final evaluation.
read -r -a EVAL_SPLITS <<< "${SPLITS:-source val}"
read -r -a EVAL_TASKS <<< "$TASKS"
for split in "${EVAL_SPLITS[@]}"; do
  case "$split" in
    source) styles=11-30 ;;
    val) styles=31-40 ;;
    test) styles=41-50 ;;
    *) echo "Unknown split: $split" >&2; exit 1 ;;
  esac
  STYLES="$styles" OUT="$OUT/$split" MANIFEST_DIR="$MANIFEST_ROOT/$split" \
    bash "$(dirname "$0")/run_corp_sr_tasks.sh"
done
"${PY:-python3}" "$(dirname "$0")/aggregate_sr.py" "$OUT" --tasks "${EVAL_TASKS[@]}" \
  --episodes "$EPISODES" --splits "${EVAL_SPLITS[@]}"
echo SPLIT_DONE
