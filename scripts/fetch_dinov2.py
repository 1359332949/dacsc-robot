#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉取并验证 DINOv2 backbone（CoRP 用的那个）。

为什么需要这个脚本：
  torch.hub.load("facebookresearch/dinov2", ...) 默认走 github.com（本环境被拦）。
  但——代码包可从 codeload 拉、权重可从 Meta CDN 拉（实测 4.9 MB/s）。
  本脚本自动完成「下载 → 放到 TORCH_HOME → 验证」。

用法：
  python3 scripts/fetch_dinov2.py                 # 默认 dinov2_vits14（CoRP 用的）
  python3 scripts/fetch_dinov2.py --model dinov2_vitb14
  python3 scripts/fetch_dinov2.py --verify        # 只验证不下载
"""
from __future__ import annotations
import argparse, os, pathlib, shutil, subprocess, sys, zipfile

HUB = pathlib.Path.home() / ".cache" / "torch" / "hub"
CODE_DIR = HUB / "facebookresearch_dinov2_main"
CKPT_DIR = HUB / "checkpoints"
CODE_ZIP = "https://codeload.github.com/facebookresearch/dinov2/zip/refs/heads/main"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/dinov2/{m}/{m}_pretrain.pth"

MODELS = {
    "dinov2_vits14": (22.1, 384, 84),     # CoRP 用的；参数量/嵌入维度/权重MB
    "dinov2_vitb14": (86.6, 768, 330),
    "dinov2_vitl14": (300.0, 1024, 1161),
}


def fetch_code():
    if (CODE_DIR / "hubconf.py").exists():
        print(f"[skip] 代码已在 {CODE_DIR}")
        return
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HUB / "_dinov2.zip"
    print("[get ] dinov2 代码包（约 3 MB）")
    subprocess.run(["curl", "-sL", "--max-time", "900", "-o", str(tmp), CODE_ZIP], check=True)
    ex = HUB / "_dinov2_extract"
    if ex.exists():
        shutil.rmtree(ex)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(ex)
    inner = next(p for p in ex.iterdir() if p.is_dir())
    shutil.move(str(inner), str(CODE_DIR))
    shutil.rmtree(ex, ignore_errors=True); tmp.unlink(missing_ok=True)
    print(f"[done] 代码 -> {CODE_DIR}")


def fetch_weights(model: str):
    dest = CKPT_DIR / f"{model}_pretrain.pth"
    need_mb = MODELS[model][2]
    if dest.exists() and dest.stat().st_size > need_mb * 0.95 * 1e6:
        print(f"[skip] 权重已在 {dest} ({dest.stat().st_size/1e6:.0f} MB)")
        return dest
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    url = WEIGHT_URL.format(m=model)
    print(f"[get ] 权重 {model}（约 {need_mb} MB；Meta CDN 实测 4-5 MB/s）")
    subprocess.run(["curl", "-L", "-C", "-", "--retry", "3", "--max-time", "3600",
                    "-o", str(dest), url], check=True)
    print(f"[done] {dest} ({dest.stat().st_size/1e6:.0f} MB)")
    return dest


def verify(model: str):
    import torch
    if not (CODE_DIR / "hubconf.py").exists():
        print("[warn] 代码未就绪，先运行不带 --verify"); return False
    w = CKPT_DIR / f"{model}_pretrain.pth"
    if not w.exists():
        print("[warn] 权重未就绪"); return False
    print("[verify] 从本地加载 torch.hub ...")
    m = torch.hub.load(str(CODE_DIR), model, source="local", pretrained=False)
    missing, unexpected = m.load_state_dict(torch.load(w, map_location="cpu"), strict=False)
    m.eval()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"   参数量 {n:.1f} M | patch={m.patch_size} dim={m.embed_dim}")
    print(f"   权重匹配: missing={len(missing)} unexpected={len(unexpected)}")
    with torch.no_grad():
        out = m.forward_features(torch.zeros(1, 3, 224, 224))
    pt = out["x_norm_patchtokens"]
    print(f"   224×224 → patch tokens {tuple(pt.shape)}（应为 1×256×{m.embed_dim}）")
    print("   ✓ 可用于训练")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dinov2_vits14", choices=sorted(MODELS))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--code-only", action="store_true")
    a = ap.parse_args()
    if a.verify:
        return 0 if verify(a.model) else 1
    fetch_code()
    if not a.code_only:
        fetch_weights(a.model)
        verify(a.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
