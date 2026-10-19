#!/usr/bin/env bash
# 建立可跑 DACSC 的 CPU/GPU 环境（幂等，可重复执行）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"

if [ ! -d "$VENV" ]; then
  echo "[1/4] 创建 venv → $VENV"
  python3 -m venv "$VENV"
fi

echo "[2/4] 安装依赖（torch / diffusers / hydra / robomimic 依赖等）"
"$VENV/bin/pip" install --quiet --disable-pip-version-check \
  torch torchvision numpy diffusers einops hydra-core omegaconf \
  zarr numcodecs dill scikit-image scikit-video psutil click threadpoolctl \
  transformers scipy tqdm termcolor h5py imageio matplotlib

echo "[3/4] 用 .pth 让三个包可导入（不安装、不改官方仓库）"
SP="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
cat > "$SP/_dacsc_paths.pth" <<EOF
$ROOT
$ROOT/repos/diffusion_policy
$ROOT/repos/robomimic
EOF
echo "     写入 $SP/_dacsc_paths.pth"

echo "[4/4] 验证"
"$VENV/bin/python" - <<'PY'
import importlib
for m in ("torch", "diffusion_policy", "robomimic", "dacsc"):
    try:
        importlib.import_module(m); print(f"   ✓ {m}")
    except Exception as e:
        print(f"   ✗ {m}: {type(e).__name__}: {e}")
PY
echo "完成。用法：$VENV/bin/python <script>"
