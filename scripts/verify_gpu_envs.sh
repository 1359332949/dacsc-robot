#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORP_PYTHON=""
ROBOCASA_PYTHON=""
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --corp-python) CORP_PYTHON="$2"; shift 2 ;;
    --robocasa-python) ROBOCASA_PYTHON="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$CORP_PYTHON" && -n "$ROBOCASA_PYTHON" ]] || { echo "both interpreters are required" >&2; exit 1; }
exec env CORP_PYTHON="$CORP_PYTHON" ROBOCASA_PYTHON="$ROBOCASA_PYTHON" \
  VERIFY_OUT="$OUT" PYTHONPATH="$ROOT" "$CORP_PYTHON" "$ROOT/scripts/verify_gpu_envs.py"
