#!/usr/bin/env bash
# Evaluate a frozen v10 checkpoint on official RoboCasa target scenes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
MODEL="${MODEL:?set MODEL=/path/to/model.pt}"
OUT="${OUT:?set OUT=/path/to/sr/output}"
MANIFEST_ROOT="${MANIFEST_ROOT:?set MANIFEST_ROOT=/path/to/frozen/target/manifests}"
EPISODES="${EPISODES:-20}"
TASKS="${TASKS:-OpenDrawer OpenCabinet TurnOffStove TurnOnSinkFaucet TurnOnElectricKettle}"

cd "$ROOT"
TASKS="$TASKS" \
EPISODES="$EPISODES" \
BLUR_SIGMA="${BLUR_SIGMA:-0}" \
OUT="$OUT" \
SPLITS=target \
SHIFT="${SHIFT:-none}" \
MANIFEST_ROOT="$MANIFEST_ROOT" \
MANIFEST_MODE="${MANIFEST_MODE:-replay}" \
TARGET_STYLES=1-10 \
PY="$PY" \
bash scripts/run_corp_sr_split.sh
