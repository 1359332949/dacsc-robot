#!/usr/bin/env bash
# 从 ModelScope 只拉 CLIP 文本编码器所需文件。
# 不使用 git clone：该仓库包含 TF/Flax 等多个大权重，整仓 LFS 会下载 10GB+。
set -euo pipefail
DEST="${1:-/hy-tmp/models/clip-vit-large-patch14}"
BASE="${DACSC_CLIP_BASE_URL:-https://www.modelscope.cn/models/AI-ModelScope/clip-vit-large-patch14/resolve/master}"
FILES=(
  model.safetensors
  config.json
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.json
  merges.txt
  preprocessor_config.json
)
mkdir -p "$DEST"
cd "$DEST"
for f in "${FILES[@]}"; do
  if [ -s "$f" ]; then
    echo "[skip] $f ($(du -h "$f" | cut -f1))"
    continue
  fi
  echo "[get ] $f"
  curl -L -C - --fail --retry 3 --max-time 3600 -o "$f" "$BASE/$f"
done
ls -lh
