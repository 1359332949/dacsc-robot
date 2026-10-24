#!/usr/bin/env bash
# 从 HuggingFace 官方镜像下载 RoboCasa kitchen assets（hf-mirror，支持断点续传）。
set -euo pipefail
BASE="${DACSC_ROBOCASA_ASSETS_BASE:-https://hf-mirror.com/datasets/robocasa/robocasa-assets/resolve/main}"
DEST="${1:-/hy-tmp/robocasa_assets_hf}"
mkdir -p "$DEST"
FILES=(textures.zip generative_textures.zip fixtures.zip objaverse.zip aigen_objs.zip)
for f in "${FILES[@]}"; do
  if [ -s "$DEST/$f" ]; then
    echo "[skip] $f $(du -h "$DEST/$f" | cut -f1)"
    continue
  fi
  echo "[get ] $f"
  curl -L -C - --fail --retry 6 --retry-delay 5 --max-time 7200 \
    -o "$DEST/$f" "$BASE/$f"
  echo "[ok  ] $f $(du -h "$DEST/$f" | cut -f1)"
done
echo "ALL_DOWNLOADS_DONE"
