#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉取 DACSC-Robot 的参考仓库（可续传、gzip 校验、失败可重试）。

为什么不用 git clone：部分环境到 github.com 的 git/HTTPS 被拦，只有
codeload / api / raw 可达，因此默认走 tarball 快照。需要 git 历史的请用
`--mode git`（前提是该环境能访问 github.com）。

用法：
  python3 scripts/fetch_repos.py --list
  python3 scripts/fetch_repos.py --group small       # 小仓库（<5MB）
  python3 scripts/fetch_repos.py --only RoboAug
  python3 scripts/fetch_repos.py --group core        # 官方基线核心
  python3 scripts/fetch_repos.py                     # 全部
"""
from __future__ import annotations
import argparse, gzip, os, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "repos"

# name, slug, branch, group, size_mb, note
MANIFEST = [
    ("XPolicyLab", "XPolicyLab/XPolicyLab", "main", "core", 50.8,
     "★★★首推：63 个机器人策略的统一评测/部署框架（Apache-2.0，2026-10-14，IROS26 Best Tool）"),
    ("RoboTwin", "RoboTwin-Platform/RoboTwin", "main", "core", 49.7,
     "★★★RoboTwin 2.0 官方仓库（MIT，2968★，ICML2026）：强域随机化数据生成+基准"),
    ("RoboDojo", "RoboDojo-Benchmark/RoboDojo", "main", "core", 0.2,
     "★RoboDojo 官方仓库（MIT，738★）：42 仿真任务，5 维含 Generalization + 18 真机任务"),
    ("RMBench", "RoboTwin-Platform/RMBench", "main", "core", 2.3,
     "RoboTwin 平台下的记忆依赖操作基准（MIT，221★）"),
    ("RoboTransfer", "HorizonRobotics/RoboTransfer", "main", "aug", 21.3,
     "几何一致视频扩散用于策略迁移的数据增广（Apache-2.0，2025-05）"),
    ("RoboManipBaselines", "isri-aist/RoboManipBaselines", "master", "il", 108.5,
     "IL 特化基线框架（BSD-2，405★）：10 个策略 mlp/sarnn/act/mt_act/DP/3D-DP/flow/mani_flow/pi0/gr00t"),
    ("RoboAug", "Open-X-Humanoid/RoboAug", "main", "small", 0.1,
     "★区域对比数据增广（CoRL2026，arXiv 2602.14032）；本仓库仅含数据生成(stage1-2)"),
    ("any4lerobot", "Tavish9/any4lerobot", "main", "small", 0.3,
     "LeRobot 数据格式转换/可视化工具"),
    ("awesome-embodied-vla-va-vln", "jonyzhang2023/awesome-embodied-vla-va-vln", "main", "small", 0.6,
     "VLA/VA/VLN 综述列表（2026-09 仍在更新）"),
    ("openpi", "robocasa-benchmark/openpi", "main", "core", 0.8,
     "官方 pi0 fork（RoboCasa365 基线之一）"),
    ("Awesome-RL-VLA", "Denghaoyuan123/Awesome-RL-VLA", "main", "small", 1.2,
     "VLA 强化学习综述（2026-05）"),
    ("RoboVIP_VDM", "RoboVIP/RoboVIP_VDM", "main", "small", 2.6,
     "多视角视频增广（arXiv 2601.05241）；仿真训练/评测未发布"),
    ("leaderboard", "robocasa-benchmark/leaderboard", "main", "core", 2.7,
     "★官方排行榜提交记录（含 DP/pi0/GR00T/WAM 全部数值）"),
    ("NICE-Scene-Surgery", "SajjadPSavoji/NICE-Scene-Surgery", "main", "small", 4.7,
     "场景编辑增广（CoRL2025 Workshop，成熟度低）"),
    ("diffusion_policy", "robocasa-benchmark/diffusion_policy", "main", "core", 11.6,
     "★★主基线：官方 Diffusion Policy fork"),
    ("mimicgen", "robocasa-benchmark/mimicgen", "main", "core", 15.6,
     "MimicGen（RoboCasa pretrain MG 数据集生成）"),
    ("lerobot-dataset-visualizer", "huggingface/lerobot-dataset-visualizer", "main", "tool", 21.6,
     "LeRobot 数据集网页可视化（阶段0 审计用）"),
    ("roboengine", "michaelyuancb/roboengine", "main", "small", 27.5,
     "★即插即用增广工具箱（IROS2025，arXiv 2503.18738）"),
    ("lerobot", "huggingface/lerobot", "main", "tool", 30.0,
     "LeRobot 官方框架（RoboCasa365 数据格式基座）"),
    ("robocasa", "robocasa/robocasa", "main", "core", 47.7,
     "★★RoboCasa365 仿真本体（任务/场景/数据下载脚本）"),
    ("Isaac-GR00T", "robocasa-benchmark/Isaac-GR00T", "main", "core", 47.8,
     "官方 GR00T N1.5 fork"),
    ("robomimic", "robocasa-benchmark/robomimic", "robocasa-dp", "core", 53.4,
     "★★ResNet18ConvFiLM 定义所在（必须 robocasa-dp 分支）"),
    ("RoboSaGA", "zheyu-zhuang/RoboSaGA", "master", "big", 680.0,
     "★显著性引导增广（CoRL2024）；内嵌 robosuite 故 680MB，建议只抓关键源码"),
    ("Awesome-WAM", "OpenMOSS/Awesome-WAM", "main", "big", 1796.0,
     "World Action Model 综述（当前 SOTA 家族，1.8GB 含论文 PDF）"),
]
BY_NAME = {m[0]: m for m in MANIFEST}


def tarball_url(slug, branch):
    return f"https://codeload.github.com/{slug}/tar.gz/refs/heads/{branch}"


CACHE = None  # 在 main/调用处初始化为 DEST/.cache


def _verify_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except Exception as exc:
        print(f"       gzip 校验失败：{exc}")
        return False


def _safe_filter(member, dest):
    """允许常规成员；跳过绝对路径符号链接（部分仓库带这类链接，如 XPolicyLab）。"""
    try:
        return tarfile.data_filter(member, dest)
    except tarfile.AbsoluteLinkError:
        print(f"       [skip] 跳过绝对路径链接：{member.name}")
        return None
    except Exception:
        return None


def _extract(tgz: Path, dest: Path):
    """解压到 dest；优先用安全过滤器，失败则退回 fully_trusted。"""
    with tarfile.open(tgz, "r:gz") as tf:
        try:
            tf.extractall(dest, filter=_safe_filter)
        except Exception as exc:
            print(f"       安全过滤器失败（{type(exc).__name__}），改用 fully_trusted")
            with tarfile.open(tgz, "r:gz") as tf2:
                tf2.extractall(dest, filter="fully_trusted")


def download(name, slug, branch, retries=3, timeout=3600):
    """下载 + 解压。tarball 缓存在 repos/.cache/<name>.tgz，重跑不会重复下载。"""
    global CACHE
    if CACHE is None:
        CACHE = DEST / ".cache"
    target = DEST / name
    if target.exists() and any(target.iterdir()):
        print(f"[skip] {name} 已存在")
        return True

    CACHE.mkdir(parents=True, exist_ok=True)
    tgz = CACHE / f"{name}.tgz"

    if not (tgz.exists() and tgz.stat().st_size > 0 and _verify_gzip(tgz)):
        ok = False
        for attempt in range(1, retries + 1):
            print(f"[get ] {name} <- {slug}@{branch} (第 {attempt} 次)")
            t0 = time.time()
            r = subprocess.run(["curl", "-sL", "--max-time", str(timeout), "-o", str(tgz),
                                "-w", "%{http_code} %{size_download}",
                                tarball_url(slug, branch)], capture_output=True, text=True)
            code, _, size = (r.stdout or "").partition(" ")
            print(f"       HTTP={code} size={size} 用时 {time.time()-t0:.0f}s")
            if tgz.exists() and tgz.stat().st_size > 0 and _verify_gzip(tgz):
                ok = True
                break
        if not ok:
            print(f"[FAIL] {name} 下载失败")
            return False
    else:
        print(f"[cache] {name} 复用已缓存的 tarball（{tgz.stat().st_size/1e6:.1f} MB）")

    tmp_ex = Path(tempfile.mkdtemp(prefix=f".{name}.x."))
    try:
        _extract(tgz, tmp_ex)
        inner = [q for q in tmp_ex.iterdir() if q.is_dir()]
        if not inner:
            print(f"[FAIL] {name} 解压结果为空")
            return False
        shutil.move(str(inner[0]), str(target))
        n = sum(1 for _ in target.rglob("*"))
        print(f"[done] {name} -> {target}  ({n} 项)")
        return True
    finally:
        shutil.rmtree(tmp_ex, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append")
    ap.add_argument("--group", action="append",
                    choices=["small", "core", "tool", "big", "aug", "il"])
    ap.add_argument("--max-size", type=float, default=None, help="只拉小于该 MB 数的仓库")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'name':<30}{'slug':<45}{'branch':<12}{'group':<8}{'MB':>8}  note")
        for n, s, b, g, mb, note in MANIFEST:
            print(f"{n:<30}{s:<45}{b:<12}{g:<8}{mb:>8.1f}  {note}")
        return

    DEST.mkdir(parents=True, exist_ok=True)
    sel = []
    for m in MANIFEST:
        n, s, b, g, mb, note = m
        if args.only and n not in args.only:
            continue
        if args.group and g not in args.group:
            continue
        if args.max_size is not None and mb > args.max_size:
            continue
        sel.append(m)
    print(f"计划拉取 {len(sel)} 个仓库，总计约 {sum(m[4] for m in sel):.1f} MB\n")
    ok = fail = 0
    for n, s, b, g, mb, note in sel:
        if download(n, s, b):
            ok += 1
        else:
            fail += 1
    print(f"\n完成：成功 {ok}，失败 {fail}")


if __name__ == "__main__":
    main()
