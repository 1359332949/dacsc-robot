#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速检查 RoboCasa365 的落盘路径、任务数和域标签完整性。

只检查 metadata，不读 parquet/video，通常几秒完成。
用法：
  python3 scripts/check_data_layout.py
  python3 scripts/check_data_layout.py --root ~/datasets/robocasa365
  python3 scripts/check_data_layout.py --root /root/autodl-tmp/datasets/robocasa365
  python3 scripts/check_data_layout.py --root /hy-tmp/datasets/robocasa365   # 恒源云
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def default_root() -> Path:
    env = os.environ.get("DACSC_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    for data_mount in (Path("/root/autodl-tmp"), Path("/hy-tmp")):
        if data_mount.exists():
            return data_mount / "datasets" / "robocasa365"
    return Path.home() / "datasets" / "robocasa365"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=default_root())
    ap.add_argument("--split", default="pretrain")
    ap.add_argument("--kind", default="atomic")
    a = ap.parse_args()

    root = a.root.expanduser().resolve()
    data = root / "v1.0" / a.split / a.kind
    print(f"数据集根目录 : {root}")
    print(f"本次检查目录 : {data}")
    if not data.is_dir():
        print("❌ 目录不存在")
        return 1

    tasks = sorted(p for p in data.iterdir() if p.is_dir())
    metas = sorted(data.rglob("ep_meta.json"))
    missing_meta = []
    bad_json = []
    missing_style = []
    missing_layout = []
    styles = set()
    layouts = set()

    for p in tasks:
        lerobot = list(p.glob("*/lerobot"))
        if not lerobot or not any(lerobot[0].iterdir()):
            missing_meta.append(str(p))

    for meta in metas:
        try:
            obj = json.loads(meta.read_text())
        except Exception:
            bad_json.append(str(meta))
            continue
        if obj.get("style_id") is None:
            missing_style.append(str(meta))
        else:
            styles.add(int(obj["style_id"]))
        if obj.get("layout_id") is None:
            missing_layout.append(str(meta))
        else:
            layouts.add(int(obj["layout_id"]))

    print(f"任务数       : {len(tasks)}")
    print(f"episode 数   : {len(metas)}")
    print(f"style 数     : {len(styles)}  ({min(styles) if styles else '-'}..{max(styles) if styles else '-'})")
    print(f"layout 数    : {len(layouts)} ({min(layouts) if layouts else '-'}..{max(layouts) if layouts else '-'})")
    print(f"缺失 style_id: {len(missing_style)}")
    print(f"缺失 layout_id:{len(missing_layout)}")
    print(f"JSON 解析失败: {len(bad_json)}")
    print(f"空/缺失 lerobot: {len(missing_meta)}")

    ok = not (missing_meta or bad_json or missing_style or missing_layout) and metas
    print("结果         :", "✅ PASS" if ok else "❌ FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
