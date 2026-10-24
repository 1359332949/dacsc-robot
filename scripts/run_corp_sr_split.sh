#!/usr/bin/env bash
# source/val/test success-rate for one model, one shift.
set -euo pipefail
OUT="${OUT:?set OUT=/path/to/out_root}"
export MODEL="${MODEL:?set MODEL=/path/to/model.pt}"
export EPISODES="${EPISODES:-10}"
export TASKS="${TASKS:-CloseFridge OpenFridge CloseDrawer OpenDrawer CloseCabinet OpenCabinet TurnOnStove TurnOffStove PickPlaceCounterToCabinet PickPlaceCabinetToCounter}"
export SHIFT="${SHIFT:-none}"
STYLES=11-30 OUT="$OUT/source" bash "$(dirname "$0")/run_corp_sr_tasks.sh"
STYLES=31-40 OUT="$OUT/val" bash "$(dirname "$0")/run_corp_sr_tasks.sh"
STYLES=41-50 OUT="$OUT/test" bash "$(dirname "$0")/run_corp_sr_tasks.sh"
echo SPLIT_DONE
