#!/usr/bin/env bash
# 在本机（Mac）打包要上传到 GPU 服务器的内容
# 用法：bash scripts/pack_for_server.sh [输出目录]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/.cache/upload}"
mkdir -p "$OUT"

echo "[1/3] 打包我们的代码（dacsc / scripts / docs / notes）"
tar -czf "$OUT/dacsc_code.tgz" -C "$ROOT" dacsc scripts docs notes
echo "      → $OUT/dacsc_code.tgz  ($(du -h "$OUT/dacsc_code.tgz" | cut -f1))"

echo "[2/3] 打包三个官方仓库（可选：服务器能 clone 就不用传）"
tar -czf "$OUT/dacsc_repos.tgz" -C "$ROOT" \
    repos/diffusion_policy repos/robomimic repos/robocasa
echo "      → $OUT/dacsc_repos.tgz  ($(du -h "$OUT/dacsc_repos.tgz" | cut -f1))"

echo "[3/3] 打包 DINOv2 代码（权重建议服务器直接下，4.9MB/s）"
if [ -f "$ROOT/.cache/dinov2/dinov2.zip" ]; then
  cp "$ROOT/.cache/dinov2/dinov2.zip" "$OUT/dinov2.zip"
  echo "      → $OUT/dinov2.zip  ($(du -h "$OUT/dinov2.zip" | cut -f1))"
fi

echo
echo "上传命令（把 <PORT>/<HOST> 换成 AutoDL 给的）："
echo "  scp -P <PORT> $OUT/dacsc_code.tgz  root@<HOST>:/root/"
echo "  scp -P <PORT> $OUT/dacsc_repos.tgz root@<HOST>:/root/    # 可选"
