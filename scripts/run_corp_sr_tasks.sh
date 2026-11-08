#!/usr/bin/env bash
# Multi-task success-rate evaluation for CoRP-RoboCasa.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PY="${CONFIG_PY:-python3}"
cd "$ROOT"
eval "$("$CONFIG_PY" -m dacsc.v10_config --shell)"
MODEL="${MODEL:?set MODEL=/path/to/model.pt}"
OUT="${OUT:?set OUT=/path/out}"
TASKS="${TASKS:-$V10_TASKS}"
STYLES="${STYLES:-$V10_SOURCE_STYLES}"
EPISODES="${EPISODES:-$V10_EPISODES}"
PY="${PY:-python3}"
SHIFT="${SHIFT:-none}"
BLUR_SIGMA="${BLUR_SIGMA:-0}"
SCENE_SPLIT="${SCENE_SPLIT:-pretrain}"
MANIFEST_DIR="${MANIFEST_DIR:?set MANIFEST_DIR=/path/to/frozen/split_manifest}"
MANIFEST_MODE="${MANIFEST_MODE:-replay}"
ACTION_STEPS="${ACTION_STEPS:-$V10_ACTION_STEPS}"
LAYOUT_ID="${LAYOUT_ID:-$V10_LAYOUT_ID}"
SPLIT_NAME="${SPLIT_NAME:-source}"
if [[ -n "${MAX_STEPS+x}" ]]; then
  echo "MAX_STEPS is unsupported; horizons come from the official task registry" >&2
  exit 2
fi
mkdir -p "$OUT"
LAYOUT_ARGS=()
if [[ "$SCENE_SPLIT" == "pretrain" ]]; then
  if [[ -z "$LAYOUT_ID" ]]; then
    echo "pretrain evaluation requires explicit LAYOUT_ID" >&2
    exit 2
  fi
  LAYOUT_ARGS=(--layout-id "$LAYOUT_ID")
fi
for T in $TASKS; do
  echo "[eval] $T styles=$STYLES shift=$SHIFT"
  FIXTURE_ARGS=()
  if [[ "$SPLIT_NAME" == "source" ]]; then
    if [[ -z "${TARGET_FIXTURES_JSON:-}" ]]; then
      echo "source evaluation requires TARGET_FIXTURES_JSON" >&2
      exit 2
    fi
    TASK_FIXTURES="$(
      TARGET_FIXTURES_JSON="$TARGET_FIXTURES_JSON" TASK_NAME="$T" \
        "$CONFIG_PY" -c \
        'import json, os; print(json.dumps(json.loads(os.environ["TARGET_FIXTURES_JSON"])[os.environ["TASK_NAME"]]))'
    )"
    FIXTURE_ARGS=(--fixture-refs-json "$TASK_FIXTURES")
  fi
  MUJOCO_GL=egl PYTHONPATH="$ROOT" "$PY" -m dacsc.run_output --file \
    --out "$OUT/$T.json" --log "$T.log" -- "$ROOT/scripts/eval_corp_robocasa_sr.py" \
    --model "$MODEL" --task "$T" --scene-split "$SCENE_SPLIT" \
    --styles "$STYLES" --episodes "$EPISODES" --action-steps "$ACTION_STEPS" \
    "${LAYOUT_ARGS[@]}" \
    "${FIXTURE_ARGS[@]}" \
    --blur-sigma "$BLUR_SIGMA" \
    --manifest-dir "$MANIFEST_DIR/$T" --manifest-mode "$MANIFEST_MODE" \
    --shift "$SHIFT" --out "$OUT/$T.json"
done
echo ALL_DONE
