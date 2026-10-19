#!/usr/bin/env bash
# 在 GPU 服务器（AutoDL）上一键装环境。**无卡模式也能跑**（数据下载/装环境不需要 GPU）。
# 用法：
#   bash setup_server.sh            # 完整安装 + 自检
#   bash setup_server.sh --no-gpu   # 无卡模式（跳过 MPS/CUDA 检查）
set -uo pipefail
ROOT="${DACSC_ROOT:-/root/dacsc-robot}"
NOGPU=0
[ "${1:-}" = "--no-gpu" ] && NOGPU=1

# 云厂商数据盘：AutoDL=/root/autodl-tmp，恒源云=/hy-tmp。
# 系统盘通常只有 20GB，pip/torch/HF 缓存必须赶到数据盘。
DATA_MOUNT=""
for d in /root/autodl-tmp /hy-tmp; do
  if [ -d "$d" ]; then DATA_MOUNT="$d"; break; fi
done
if [ -n "$DATA_MOUNT" ]; then
  mkdir -p "$DATA_MOUNT/cache/pip" "$DATA_MOUNT/cache/torch"            "$DATA_MOUNT/cache/huggingface" "$DATA_MOUNT/tmp" "$DATA_MOUNT/wandb"
  export PIP_CACHE_DIR="$DATA_MOUNT/cache/pip"
  export TORCH_HOME="$DATA_MOUNT/cache/torch"
  export HF_HOME="$DATA_MOUNT/cache/huggingface"
  export XDG_CACHE_HOME="$DATA_MOUNT/cache"
  export TMPDIR="$DATA_MOUNT/tmp"
  export WANDB_DIR="$DATA_MOUNT/wandb"
  echo "数据盘: $DATA_MOUNT（缓存已重定向）"
else
  echo "⚠ 未发现数据盘，缓存仍使用系统盘"
fi

echo "=============================================="
echo " DACSC-Robot 服务器环境安装   root=$ROOT"
echo "=============================================="

echo "[1/6] 检查基础环境"
python3 -V
python3 - <<'PY'
import torch
print("  torch:", torch.__version__)
print("  CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))
    print("  显存: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory/1e9))
PY

echo
echo "[2/6] 安装依赖（跳过已装的 torch）"
# 与 torch 2.9+ / CUDA 12.8 镜像匹配（本机已验证的组合）
python3 -m pip install --quiet --no-cache-dir --disable-pip-version-check \
  "diffusers==0.41.0" "transformers==5.20.0" "accelerate==1.15.0" \
  einops hydra-core omegaconf zarr numcodecs dill \
  scikit-image scikit-video psutil click threadpoolctl scipy \
  tqdm termcolor h5py imageio matplotlib pandas wandb \
  || echo "  ⚠ 部分依赖安装失败，请检查网络（可用 pip 国内源）"

echo
echo "[3/6] 让三个包可导入（.pth，不安装、不改官方仓库）"
SP=$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || python3 -c 'import site; print(site.getsitepackages()[0])')
cat > "$SP/_dacsc_paths.pth" <<EOF
$ROOT
$ROOT/repos/diffusion_policy
$ROOT/repos/robomimic
$ROOT/repos/robocasa
EOF
echo "     写入 $SP/_dacsc_paths.pth"

echo
echo "[4/6] 拉取 DINOv2（代码 + ViT-S/14 权重，约 87MB）"
cd "$ROOT" && python3 scripts/fetch_dinov2.py 2>&1 | tail -6

echo
echo "[5/6] 自检（4 个测试脚本）"
cd "$ROOT"
for t in test_dg_methods test_gating test_dinov2_policy test_domain_labels; do
  printf "  %-24s " "$t"
  if timeout 600 python3 "scripts/$t.py" > "/tmp/$t.log" 2>&1; then
    echo "✓ 通过"
  else
    echo "✗ 失败（日志 /tmp/$t.log）"
  fi
done

echo
echo "[6/6] 合成数据端到端训练（$([ $NOGPU -eq 1 ] && echo 'CPU' || echo 'GPU')）"
cd "$ROOT"
SMOKE_STEPS=8 SMOKE_BS=8 timeout 1200 python3 scripts/train_smoke_test.py 2>&1 | tail -12

echo
echo "=============================================="
echo " 完成。下一步："
echo "   python3 scripts/fetch_robocasa365_modelscope.py --list --split pretrain --kind atomic"
echo "   python3 scripts/fetch_robocasa365_modelscope.py --preset atomic-all --audit"
echo "=============================================="
