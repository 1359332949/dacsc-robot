#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: download_gpu_wheelhouse.sh DIRECTORY}"
PYTHON="${PYTHON:-python3}"
WORKERS="${WHEELHOUSE_WORKERS:-16}"
CHUNK_SIZE="${WHEELHOUSE_CHUNK_SIZE:-33554432}"
if [[ -e "$OUT" && ! -d "$OUT" ]]; then
  echo "Refusing non-directory output path: $OUT" >&2
  exit 1
fi
if [[ -d "$OUT" && -n "$(ls -A "$OUT")" && ! -f "$OUT/.wheelhouse.lock" ]]; then
  echo "Refusing non-wheelhouse directory: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"
exec "$PYTHON" "$ROOT/scripts/download_gpu_wheelhouse.py" \
  --root "$ROOT" \
  --out "$OUT" \
  --pypi "${PYPI:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
  --pytorch "${PYTORCH:-https://download.pytorch.org/whl/cu128}" \
  --chunk-size "$CHUNK_SIZE" \
  --workers "$WORKERS"
