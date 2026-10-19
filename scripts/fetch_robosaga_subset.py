#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只抓 RoboSaGA 的方法源码 + 集成脚本 + 配置（约 200KB），跳过 680MB 的内嵌 robosuite。

用法：
  python3 scripts/fetch_robosaga_subset.py                 # 源码+配置（约 200KB）
  python3 scripts/fetch_robosaga_subset.py --with-backgrounds   # 额外抓 backgrounds.zip (27.5MB)

落盘位置：repos/RoboSaGA/<原仓库相对路径>
"""
from __future__ import annotations
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "repos" / "RoboSaGA"
RAW = "https://raw.githubusercontent.com/zheyu-zhuang/RoboSaGA/master"

CORE = [
    # 核心方法实现
    "robosaga/robosaga/__init__.py",
    "robosaga/robosaga/saliency_guided_augmentation.py",   # ★ 方法主体
    "robosaga/robosaga/background_randomizer.py",
    "robosaga/robosaga/buffer_manager.py",
    "robosaga/robosaga/custom_group_norm.py",
    "robosaga/robosaga/fullgrad.py",                       # 显著性全梯度
    "robosaga/robosaga/resnet.py",
    "robosaga/robosaga/soda.py",                           # SODA 对比方法
    # robomimic 侧集成与评测
    "robomimic/robomimic/scripts/generate_robosaga_configs.py",
    "robomimic/robomimic/scripts/train_soda.py",
    "robomimic/robomimic/scripts/eval_visual_domain_shifts.py",  # ★ 视觉域偏移评测协议
    # diffusion policy 侧集成
    "diffusion_policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy_soda.py",
    # 环境与说明
    "README.md", "environment.yaml", "install_mujoco200.sh",
]

CONFIG_GLOBS = [
    "diffusion_policy/diffusion_policy/config/train_diffusion_unet_hybrid_can_image_{saga,soda,overlay}.yaml",
    "diffusion_policy/diffusion_policy/config/train_diffusion_unet_hybrid_lift_image_{saga,soda,overlay}.yaml",
    "diffusion_policy/diffusion_policy/config/train_diffusion_unet_hybrid_square_image_{saga,soda,overlay}.yaml",
    "diffusion_policy/diffusion_policy/config/train_diffusion_unet_hybrid_transport_image_{saga,soda,overlay}.yaml",
]
# robosaga_configs：3 任务 × (baseline, baseline_jitter, overlay, saga, soda) × (bc, bc_rnn)
for task in ("can", "lift", "square"):
    for pol in ("bc", "bc_rnn"):
        for aug in ("baseline", "baseline_jitter", "overlay", "saga", "soda"):
            CONFIG_GLOBS.append(
                f"robomimic/robomimic/robosaga_configs/{task}/{pol}_{aug}.json")


def fetch(rel: str, retries: int = 3) -> bool:
    out = DEST / rel
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {rel}")
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        r = subprocess.run(["curl", "-sL", "--max-time", "180", "-o", str(out),
                            "-w", "%{http_code}", f"{RAW}/{rel}"], capture_output=True, text=True)
        code = (r.stdout or "").strip()
        if code == "200" and out.stat().st_size > 0:
            print(f"[ok  ] {rel}  ({out.stat().st_size} bytes)")
            return True
        time.sleep(2)
    print(f"[FAIL] {rel} (HTTP={code})")
    if out.exists():
        out.unlink()
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-backgrounds", action="store_true",
                    help="额外抓 backgrounds.zip（27.5MB，OOD 背景图池）")
    args = ap.parse_args()

    files = list(CORE)
    for g in CONFIG_GLOBS:
        if "{" in g:
            head, _, tail = g.partition("{")
            for opt in tail.rstrip("}").split(","):
                files.append(head + opt)
        else:
            files.append(g)
    if args.with_backgrounds:
        files.append("backgrounds.zip")

    print(f"目标目录：{DEST}\n计划抓取 {len(files)} 个文件\n")
    ok = sum(1 for f in files if fetch(f))
    print(f"\n完成：{ok}/{len(files)}")
    print("注意：本子集不含内嵌 robosuite/robomimic，运行前需自行按其 environment.yaml 建环境。")


if __name__ == "__main__":
    main()
