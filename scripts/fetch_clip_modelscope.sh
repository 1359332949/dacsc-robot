#!/usr/bin/env bash
# 从 ModelScope 拉取 CLIP 文本编码器（RoboCasa 语言嵌入需要）。
# 只拉 model.safetensors + tokenizer/config，避免同时拉 TF/Flax 大文件。
set -euo pipefail
DEST="${1:-/hy-tmp/models/clip-vit-large-patch14}"
mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 https://www.modelscope.cn/AI-ModelScope/clip-vit-large-patch14.git "$DEST"
fi
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "请先安装 git-lfs：apt-get update && apt-get install -y git-lfs"
  exit 1
fi
cd "$DEST"
git lfs pull --include="model.safetensors,tokenizer.json,tokenizer_config.json,special_tokens_map.json,vocab.json,merges.txt,config.json"
ls -lh model.safetensors
