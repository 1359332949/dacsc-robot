#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV:-uv}"
PYTHON="${PYTHON:-3.12}"
PLATFORM="${PLATFORM:-x86_64-manylinux_2_28}"
PYPI="${PYPI:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH="${PYTORCH:-https://download.pytorch.org/whl/cu128}"

for name in corp robocasa; do
  "$UV" pip compile "$ROOT/environment/$name.in" \
    --python-version "$PYTHON" \
    --python-platform "$PLATFORM" \
    --index "$PYPI" \
    --index "$PYTORCH" \
    --index-strategy unsafe-best-match \
    --generate-hashes \
    --no-annotate \
    --no-header \
    --output-file "$ROOT/environment/$name.lock.txt"
done
echo "Wrote environment/corp.lock.txt and environment/robocasa.lock.txt"
