#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 **ModelScope** 下载 RoboCasa365（推荐；实测 22.6 MB/s，比 Box 快 600 倍）。

数据集：https://www.modelscope.cn/datasets/nv-community/robocasa365-datasets
许可：CC-BY-4.0（与官方一致）
内容：官方 RoboCasa365 v1.0 的**人类示范**（不含 MimicGen），
      每个包 = 一个任务的 lerobot.tar，**内含 extras/*/ep_meta.json（style_id / layout_id）**。

实测体积：
    pretrain/atomic      61 包   5.63 GB   ← 原子任务，短，最适合消融
    pretrain/composite  240 包 107.81 GB
    target/atomic        18 包   9.17 GB
    target/composite     32 包  52.37 GB

用法
----
# 看清单
python3 scripts/fetch_robocasa365_modelscope.py --list --split pretrain --kind atomic

# 探针：只下 2 个任务
python3 scripts/fetch_robocasa365_modelscope.py --preset probe --root /root/autodl-tmp/datasets/robocasa365

# 下全部 atomic 预训练（5.63 GB，约 4 分钟）
python3 scripts/fetch_robocasa365_modelscope.py --preset atomic-all

# 下完自动审计
python3 scripts/fetch_robocasa365_modelscope.py --preset atomic-all --audit

路径约定
--------
- 显式指定：--root /你的/数据盘/robocasa365
- 统一环境变量：DACSC_DATA_ROOT=/你的/数据盘/robocasa365
- 本机默认：~/datasets/robocasa365
- AutoDL 默认：/root/autodl-tmp/datasets/robocasa365（数据盘，避免写满系统盘）
- 恒源云默认：/hy-tmp/datasets/robocasa365（数据盘）
"""
from __future__ import annotations

import argparse, json, os, re, shutil, subprocess, sys, tarfile, time
from pathlib import Path

REPO = "nv-community/robocasa365-datasets"
BASE = f"https://www.modelscope.cn/api/v1/datasets/{REPO}/repo/tree"
RESOLVE = f"https://www.modelscope.cn/datasets/{REPO}/resolve/master/"
def default_root() -> Path:
    """数据集根目录（其下是 v1.0/...）。

    优先使用 DACSC_DATA_ROOT；AutoDL 上放到数据盘，本机放到 ~/datasets。
    代码仓库和数据集分开，避免 5GB+ 数据被误打包或误提交。
    """
    env = os.environ.get("DACSC_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    # 云厂商数据盘：AutoDL 与恒源云路径不同
    for data_mount in (Path("/root/autodl-tmp"), Path("/hy-tmp")):
        if data_mount.exists():
            return data_mount / "datasets" / "robocasa365"
    return Path.home() / "datasets" / "robocasa365"

DEFAULT_ROOT = default_root()

PROBE_TASKS = ["OpenDrawer", "PickPlaceCounterToCabinet"]


def list_tree(root: str, page_size: int = 1000, retries: int = 3) -> list[dict]:
    url = f"{BASE}?Revision=master&Root={root}&Recursive=true&PageNumber=1&PageSize={page_size}"
    for i in range(retries):
        p = subprocess.run(["curl", "-sL", "--max-time", "180", url], capture_output=True)
        try:
            d = json.loads(p.stdout)
            return d.get("Data", d).get("Files") or []
        except Exception:
            time.sleep(3)
    return []


def collect(split: str, kind: str) -> list[dict]:
    """返回该类别下所有 .tar 的 {path, size}。"""
    files = list_tree(f"{split}/{kind}")
    return [f for f in files if str(f.get("Path", "")).endswith(".tar")]


def download(path: str, dest: Path, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = RESOLVE + path
    for i in range(1, retries + 1):
        t0 = time.time()
        r = subprocess.run(["curl", "-sL", "--fail", "-C", "-", "--max-time", "3600",
                            "-o", str(dest), "-w", "%{http_code} %{size_download}", url],
                           capture_output=True, text=True)
        code, _, size = (r.stdout or "").partition(" ")
        mb = int(size or 0) / 1e6
        if code in ("200", "206") and dest.exists() and dest.stat().st_size > 0:
            print(f"      ✓ {mb:.1f} MB  {time.time()-t0:.1f}s"
                  f"  ({mb/max(time.time()-t0,1e-6):.0f} MB/s)")
            return True
        print(f"      第 {i} 次失败 HTTP={code}")
    return False


def extract(tar_path: Path, extract_dir: Path) -> bool:
    """tar 内含顶层 lerobot/，解压到 extract_dir 后得到 extract_dir/lerobot/"""
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path, "r:") as tf:
            tf.extractall(extract_dir, filter="data")
        return True
    except Exception as exc:
        print(f"      解压失败：{exc}")
        try:
            with tarfile.open(tar_path, "r:") as tf:
                tf.extractall(extract_dir, filter="fully_trusted")
            return True
        except Exception as exc2:
            print(f"      仍失败：{exc2}")
            return False


def main():
    ap = argparse.ArgumentParser(description="从 ModelScope 下载 RoboCasa365")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    ap.add_argument("--kind", default="atomic", choices=["atomic", "composite"])
    ap.add_argument("--tasks", nargs="+")
    ap.add_argument("--preset", choices=["probe", "atomic-all", "style-check"])
    ap.add_argument("--list", action="store_true", help="只列清单")
    ap.add_argument("--audit", action="store_true", help="下完自动审计")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.preset == "probe":
        tasks, split, kind = PROBE_TASKS, "pretrain", "atomic"
    elif a.preset == "atomic-all":
        tasks, split, kind = None, "pretrain", "atomic"
    elif a.preset == "style-check":
        tasks, split, kind = PROBE_TASKS, "pretrain", "atomic"
    else:
        tasks, split, kind = a.tasks, a.split, a.kind

    print(f"数据集: {REPO}")
    print(f"根目录: {a.root}  |  split={split}  kind={kind}")
    print("拉取清单 ...")
    items = collect(split, kind)
    if tasks:
        want = set(tasks)
        items = [f for f in items if f["Path"].split("/")[2] in want]
    total_gb = sum((f.get("Size") or 0) for f in items) / 1e9
    print(f"匹配到 {len(items)} 个包，合计 {total_gb:.2f} GB\n")

    if a.list:
        for f in items:
            print(f"  {(f.get('Size') or 0)/1e6:>8.1f} MB  {f['Path']}")
        return 0
    if not items:
        print("没有匹配的包"); return 1

    cache = Path(os.environ.get("TMPDIR", "/tmp")) / "rc365_ms_cache"
    cache.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = 0
    for n, f in enumerate(items, 1):
        rel = f["Path"]                     # pretrain/atomic/<Task>/<date>/lerobot.tar
        parts = rel.split("/")
        task, date = parts[2], parts[3]
        # ★ 与官方 get_ds_meta 期望的布局一致：<root>/v1.0/<split>/<kind>/<Task>/<date>/lerobot
        target_dir = a.root / "v1.0" / split / kind / task / date
        lerobot_dir = target_dir / "lerobot"
        print(f"[{n}/{len(items)}] {task} ({date})")
        if lerobot_dir.exists() and any(lerobot_dir.iterdir()) and not a.force:
            print("      [skip] 已存在")
            skip += 1
            continue
        tar_path = cache / f"{task}__{date}.tar"
        if not (tar_path.exists() and tar_path.stat().st_size > 0):
            if not download(rel, tar_path):
                fail += 1; continue
        else:
            print(f"      [cache] 复用 {tar_path.stat().st_size/1e6:.1f} MB")
        if extract(tar_path, target_dir):
            n_files = sum(1 for _ in lerobot_dir.rglob("*"))
            print(f"      ✓ 解压到 {lerobot_dir}  ({n_files} 项)")
            ok += 1
            try: tar_path.unlink()
            except Exception: pass
        else:
            fail += 1

    print(f"\n完成：成功 {ok}，跳过 {skip}，失败 {fail}")
    print(f"数据位置：{a.root}/v1.0/{split}/{kind}/")

    if a.audit and ok + skip:
        aud = Path(__file__).parent / "audit_dataset.py"
        if aud.exists():
            print("\n=== 自动审计 ===")
            subprocess.run([sys.executable, str(aud), "--dataset-root", str(a.root / "v1.0"),
                            "--expect-group", split, "--out", str(Path.cwd() / "audit" / split)])
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
