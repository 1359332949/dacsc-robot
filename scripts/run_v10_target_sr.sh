#!/usr/bin/env bash
# Evaluate a frozen v10 checkpoint on official RoboCasa target scenes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
CONFIG_PY="${CONFIG_PY:-python3}"
MODEL="${MODEL:?set MODEL=/path/to/model.pt}"
OUT="${OUT:?set OUT=/path/to/sr/output}"
MANIFEST_ROOT="${MANIFEST_ROOT:?set MANIFEST_ROOT=/path/to/frozen/target/manifests}"

cd "$ROOT"
eval "$("$CONFIG_PY" -m dacsc.v10_config --shell)"
EPISODES="${EPISODES:-$V10_EPISODES}"
TASKS="${TASKS:-$V10_TASKS}"
TASKS="$TASKS" \
EPISODES="$EPISODES" \
ACTION_STEPS="${ACTION_STEPS:-$V10_ACTION_STEPS}" \
BLUR_SIGMA="${BLUR_SIGMA:-0}" \
OUT="$OUT" \
SPLITS=target \
SHIFT="${SHIFT:-none}" \
MANIFEST_ROOT="$MANIFEST_ROOT" \
MANIFEST_MODE="${MANIFEST_MODE:-replay}" \
TARGET_STYLES="$V10_TARGET_STYLES" \
PY="$PY" \
bash scripts/run_corp_sr_split.sh
