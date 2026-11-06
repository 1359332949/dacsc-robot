#!/usr/bin/env bash
# Multi-task success-rate evaluation for CoRP-RoboCasa.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:?set MODEL=/path/to/model.pt}"
OUT="${OUT:?set OUT=/path/out}"
TASKS="${TASKS:-CloseFridge OpenFridge CloseDrawer OpenDrawer CloseCabinet OpenCabinet TurnOnStove TurnOffStove PickPlaceCounterToCabinet PickPlaceCabinetToCounter}"
STYLES="${STYLES:-11-50}"
EPISODES="${EPISODES:-10}"
PY="${PY:-python3}"
SHIFT="${SHIFT:-none}"
SCENE_SPLIT="${SCENE_SPLIT:-pretrain}"
MANIFEST_DIR="${MANIFEST_DIR:?set MANIFEST_DIR=/path/to/frozen/split_manifest}"
MANIFEST_MODE="${MANIFEST_MODE:-replay}"
if [[ -n "${MAX_STEPS+x}" ]]; then
  echo "MAX_STEPS is unsupported; horizons come from the official task registry" >&2
  exit 2
fi
mkdir -p "$OUT"
for T in $TASKS; do
  echo "[eval] $T styles=$STYLES shift=$SHIFT"
  MUJOCO_GL=egl PYTHONPATH="$ROOT" "$PY" -m dacsc.run_output --file \
    --out "$OUT/$T.json" --log "$T.log" -- "$ROOT/scripts/eval_corp_robocasa_sr.py" \
    --model "$MODEL" --task "$T" --scene-split "$SCENE_SPLIT" \
    --styles "$STYLES" --episodes "$EPISODES" \
    --manifest-dir "$MANIFEST_DIR/$T" --manifest-mode "$MANIFEST_MODE" \
    --shift "$SHIFT" --out "$OUT/$T.json"
done
echo ALL_DONE
