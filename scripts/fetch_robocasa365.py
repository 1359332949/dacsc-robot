#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RoboCasa365 数据集下载器（DACSC-Robot 专用）

复刻官方 `robocasa.scripts.download_datasets` 的路径逻辑，但：
  - 不需要安装 robocasa（直接解析 dataset_registry.py）
  - 支持 --dryrun 先看清单、--preset 按实验需要精选、断点续传、下载后自动审计

官方逻辑（已核对 repos/robocasa/robocasa/scripts/download_datasets.py）：
  ds_path      = <root>/v1.0/<folder>/lerobot          # folder 来自 registry 的 human_path / mg_path
  base         = <root>/v1.0
  tar_key      = <folder去掉v1.0前缀>/lerobot.tar       # 即 box_links_ds.json 的 key
  download_url = https://utexas.box.com/shared/static/<link_id>.tar
  解压目标      = <root>/v1.0/<folder>/                 # tar 内含顶层 lerobot/ 目录

用法
----
# 1) 先看清单（不发请求）
python3 scripts/fetch_robocasa365.py --dryrun --preset probe

# 2) 探针：只下 2 个任务，量一下大小和耗时
python3 scripts/fetch_robocasa365.py --preset probe --root /data/robocasa365

# 3) 风格覆盖检查用（20 个 atomic 任务）
python3 scripts/fetch_robocasa365.py --preset style-check --root /data/robocasa365 --audit

# 4) 自定义
python3 scripts/fetch_robocasa365.py --tasks PickPlaceCounterToCabinet OpenDrawer \
        --split pretrain --source human --root /data/robocasa365

# 5) 顺带下场景资产（~10GB）
python3 scripts/fetch_robocasa365.py --assets-only --root /data/robocasa365
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ROBOCASA_REPO = REPO / "repos" / "robocasa"
REGISTRY = ROBOCASA_REPO / "robocasa" / "utils" / "dataset_registry.py"
BOX_DS_JSON = ROBOCASA_REPO / "robocasa" / "models" / "assets" / "box_links" / "box_links_ds.json"
BOX_ASSETS_JSON = ROBOCASA_REPO / "robocasa" / "models" / "assets" / "box_links" / "box_links_assets.json"

# 预设：按实验需要精选任务，控制磁盘占用
PRESETS = {
    "probe": {
        "tasks": ["PickPlaceCounterToCabinet", "OpenDrawer"],
        "split": ["pretrain"], "source": ["human"],
        "note": "探针：量大小与耗时，不动大盘",
    },
    "style-check": {
        "tasks": ["PickPlaceCounterToCabinet", "OpenDrawer", "CloseDrawer", "OpenCabinet",
                  "CloseCabinet", "TurnOnSinkFaucet", "TurnOnMicrowave", "OpenMicrowave",
                  "CloseMicrowave", "PickPlaceSinkToCounter", "PickPlaceCounterToSink",
                  "TurnOnStove", "TurnOffStove", "OpenBlenderLid", "CloseBlenderLid",
                  "PickPlaceCounterToStove", "PickPlaceStoveToCounter", "OpenToasterOvenDoor",
                  "CloseToasterOvenDoor", "TurnOnElectricKettle"],
        "split": ["pretrain"], "source": ["human"],
        "note": "20 个 atomic 任务，用于阶段 0 审计 style 覆盖",
    },
    "target-eval": {
        "tasks": None, "split": ["target"], "source": ["human"],
        "note": "官方 50 个 target 任务（二次验证用）",
    },
}


def _balanced(text: str, open_pos: int) -> str:
    """从 open_pos 的 '(' 开始，返回括号配平的子串（含两端括号）。"""
    depth = 0
    for k in range(open_pos, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return text[open_pos:k + 1]
    return text[open_pos:]


def parse_registry():
    """从 dataset_registry.py 解析每个任务的路径配置与 horizon。

    注意：registry 用的是 `dict(...)` 调用而非字面量，ast.literal_eval 无法求值，
    因此改用括号配平 + 正则提取。
    """
    src = REGISTRY.read_text(encoding="utf-8")
    out = {}
    for block in ("ATOMIC_TASK_DATASETS", "COMPOSITE_TASK_DATASETS"):
        i = src.find(f"{block} = OrderedDict(")
        if i < 0:
            continue
        j = src.find("\n)\n", i)
        body = src[i:j]
        for m in re.finditer(r"^\s{4}([A-Za-z][A-Za-z0-9_]*)=dict\(", body, re.M):
            name = m.group(1)
            task_block = _balanced(body, m.end() - 1)
            cfg = {}
            # 逐个 split 子块（pretrain / target / real）
            for sm in re.finditer(r"(pretrain|target|real)=dict\(", task_block):
                split = sm.group(1)
                sub = _balanced(task_block, sm.end() - 1)
                paths = dict(re.findall(r'(\w+_path)="([^"]+)"', sub))
                if paths:
                    cfg[split] = paths
            # 任务级 horizon（缩进 8 空格，避免抓到子块里的）
            hm = re.search(r"^\s{8}horizon=(\d+),", task_block, re.M)
            if hm:
                cfg["horizon"] = int(hm.group(1))
            if cfg:
                out[name] = cfg
    return out


def box_url(shared_url: str, ext="tar") -> str:
    sid = shared_url.rstrip("/").split("/")[-1]
    base = shared_url.split("/s/")[0]
    return f"{base}/shared/static/{sid}.{ext}"


def build_plan(reg, tasks, splits, sources, root: Path, box):
    """产出 [(task, split, source, tar_key, url, extract_dir, target_dir)]"""
    plan = []
    for task, cfg in reg.items():
        if tasks is not None and task not in tasks:
            continue
        for split in splits:
            for source in sources:
                key = {"human": "human_path", "mg": "mg_path",
                       "mg_5x5": "mg_5x5_path", "mg_5x1": "mg_5x1_path"}[source]
                folder = cfg.get(split, {}).get(key)
                if not folder:
                    continue
                fname = "lerobot"
                tar_key = str(Path(folder).relative_to("v1.0") / f"{fname}.tar")
                if tar_key not in box:
                    plan.append((task, split, source, tar_key, None, None, None))
                    continue
                url = box_url(box[tar_key], "tar")
                extract_dir = root / folder            # <root>/v1.0/...
                target_dir = extract_dir / fname       # <root>/v1.0/.../lerobot
                plan.append((task, split, source, tar_key, url, extract_dir, target_dir))
    return plan


def download_one(url, dest_tar: Path, retries=3, timeout=7200) -> bool:
    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    for i in range(1, retries + 1):
        print(f"      下载第 {i}/{retries} 次：{dest_tar.name}")
        t0 = time.time()
        r = subprocess.run(["curl", "-L", "--fail", "--retry", "2", "--retry-delay", "5",
                            "-C", "-", "--max-time", str(timeout),
                            "-o", str(dest_tar), "-w", "%{http_code} %{size_download}",
                            url], capture_output=True, text=True)
        code, _, size = (r.stdout or "").partition(" ")
        mb = int(size or 0) / 1e6
        print(f"        HTTP={code} {mb:.1f} MB  用时 {time.time()-t0:.0f}s")
        if code in ("200", "206") and dest_tar.exists() and dest_tar.stat().st_size > 0:
            try:
                with tarfile.open(dest_tar, "r:") as tf:   # 校验 tar 完整性
                    tf.getnames()[:1]
                return True
            except Exception as exc:
                print(f"        tar 校验失败：{exc}")
    return False


def extract_one(tar_path: Path, extract_dir: Path) -> bool:
    """解压到 extract_dir；tar 内含顶层 lerobot/ 目录。"""
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path, "r:") as tf:
            members = tf.getmembers()
            bad = [m for m in members if m.name.startswith("/") or ".." in Path(m.name).parts]
            if bad:
                print(f"        [warn] 跳过 {len(bad)} 个可疑成员")
                members = [m for m in members if m not in bad]
            tf.extractall(extract_dir, members=members, filter="data")
        return True
    except Exception as exc:
        print(f"        解压失败（尝试 fully_trusted）：{exc}")
        try:
            with tarfile.open(tar_path, "r:") as tf:
                tf.extractall(extract_dir, filter="fully_trusted")
            return True
        except Exception as exc2:
            print(f"        解压仍失败：{exc2}")
            return False


def download_assets(root: Path, retries=3):
    """下载厨房场景资产（官方警告约 10GB）。"""
    if not BOX_ASSETS_JSON.exists():
        print(f"[error] 找不到 {BOX_ASSETS_JSON}")
        return False
    assets = json.loads(BOX_ASSETS_JSON.read_text(encoding="utf-8"))
    cache = Path(os.environ.get("TMPDIR", "/tmp")) / "robocasa_assets_cache"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"资产包 {len(assets)} 个（官方提示约 10 GB）→ 缓存 {cache}")
    ok = 0
    for name, shared in assets.items():
        if name in ("aigen_objs",):
            pass
        url = box_url(shared, "zip")
        z = cache / f"{name}.zip"
        if z.exists() and z.stat().st_size > 0:
            print(f"[skip] {name} 已存在（{z.stat().st_size/1e6:.0f} MB）")
            ok += 1
            continue
        print(f"[get ] {name}  ({url})")
        r = subprocess.run(["curl", "-L", "--fail", "--retry", "2", "-C", "-",
                            "--max-time", "7200", "-o", str(z), "-w", "%{http_code} %{size_download}",
                            url], capture_output=True, text=True)
        code, _, size = (r.stdout or "").partition(" ")
        if code in ("200", "206"):
            print(f"        OK {int(size or 0)/1e6:.0f} MB")
            ok += 1
        else:
            print(f"        失败 HTTP={code}")
    print(f"\n资产下载完成 {ok}/{len(assets)}。解压请执行官方脚本：")
    print("  python -m robocasa.scripts.download_kitchen_assets   # 需要能访问 Box 的网络")
    print(f"  （或手动把 {cache}/*.zip 解压到 robocasa/models/assets/）")
    return ok > 0


def main():
    ap = argparse.ArgumentParser(description="RoboCasa365 数据集下载 + 自动审计")
    ap.add_argument("--root", type=Path, default=Path("data/robocasa365"),
                    help="数据集根目录（对应官方 DATASET_BASE_PATH）")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--split", nargs="+", default=None, choices=["pretrain", "target"])
    ap.add_argument("--source", nargs="+", default=None,
                    choices=["human", "mg", "mg_5x5", "mg_5x1"])
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--audit", action="store_true", help="下载完成后自动跑审计")
    ap.add_argument("--assets-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="已有数据也重下")
    args = ap.parse_args()

    for p, label in [(REGISTRY, "dataset_registry.py"), (BOX_DS_JSON, "box_links_ds.json")]:
        if not p.exists():
            print(f"[error] 找不到 {label}：{p}\n  请先运行 scripts/fetch_repos.py --only robocasa")
            return 1

    if args.assets_only:
        return 0 if download_assets(args.root) else 1

    reg = parse_registry()
    box = json.loads(BOX_DS_JSON.read_text(encoding="utf-8"))

    tasks, splits, sources = args.tasks, args.split, args.source
    if args.preset:
        P = PRESETS[args.preset]
        tasks = tasks or P["tasks"]
        splits = splits or P["split"]
        sources = sources or P["source"]
    tasks = tasks if tasks is not None else list(reg.keys())
    splits = splits or ["pretrain"]
    sources = sources or ["human"]

    plan = build_plan(reg, set(tasks) if tasks else None, splits, sources, args.root, box)
    missing = [p for p in plan if p[4] is None]
    plan = [p for p in plan if p[4] is not None]

    print(f"仓库: {ROBOCASA_REPO}")
    print(f"根目录: {args.root}")
    print(f"任务 {len(set(p[0] for p in plan))} 个 | split={splits} | source={sources}")
    print(f"待下载数据包: {len(plan)} 个"
          + (f"（另有 {len(missing)} 个无 Box 链接，已跳过）" if missing else ""))
    if args.preset:
        print(f"预设说明: {PRESETS[args.preset]['note']}")
    print()

    if args.dryrun:
        print(f"{'任务':<38}{'split':<10}{'source':<9}{'box key'}")
        print("-" * 110)
        for task, split, source, tar_key, url, ex, tg in plan:
            print(f"{task:<38}{split:<10}{source:<9}{tar_key}")
        print(f"\n合计 {len(plan)} 个包。确认后去掉 --dryrun 即开始下载。")
        print("提示：官方未公布各包体积，建议先跑 --preset probe 实测单个任务大小。")
        return 0

    ok = skip = fail = 0
    for n, (task, split, source, tar_key, url, extract_dir, target_dir) in enumerate(plan, 1):
        print(f"[{n}/{len(plan)}] {task} | {split} | {source}")
        if target_dir.exists() and any(target_dir.iterdir()) and not args.force:
            print(f"      [skip] 已存在 {target_dir}")
            skip += 1
            continue
        cache = Path(os.environ.get("TMPDIR", "/tmp")) / "robocasa365_cache"
        cache.mkdir(parents=True, exist_ok=True)
        tar_path = cache / tar_key.replace("/", "__")
        if not (tar_path.exists() and tar_path.stat().st_size > 0):
            if not download_one(url, tar_path):
                print(f"      [FAIL] {tar_key}")
                fail += 1
                continue
        else:
            print(f"      [cache] 复用 {tar_path.name}（{tar_path.stat().st_size/1e6:.0f} MB）")
        if extract_one(tar_path, extract_dir):
            n_files = sum(1 for _ in target_dir.rglob("*"))
            print(f"      [done] {target_dir}  ({n_files} 项)")
            ok += 1
            tar_path.unlink(missing_ok=True)
        else:
            fail += 1
        print()

    print(f"完成：成功 {ok}，跳过 {skip}，失败 {fail}")
    if fail:
        print("重跑本命令即可续传（已成功的会跳过）。")

    if args.audit and ok + skip > 0:
        audit_py = HERE / "audit_dataset.py"
        if audit_py.exists():
            print("\n=== 自动审计 ===")
            cmd = [sys.executable, str(audit_py), "--dataset-root", str(args.root),
                   "--expect-group", splits[0], "--out", str(REPO / "audit" / "pretrain")]
            print("  $", " ".join(cmd))
            subprocess.run(cmd)
        else:
            print(f"[warn] 未找到 {audit_py}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
