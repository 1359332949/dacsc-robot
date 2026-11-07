#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fetch the four official source checkouts used by the active pipeline."""
from __future__ import annotations
import argparse
import gzip
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "repos"

# name, slug, revision, size_mb
MANIFEST = [
    ("CoRP", "corp-policy/CoRP", "52445740aa9517ae396cc8fb14bae8fe5922f118", 220.0),
    ("robocasa", "robocasa/robocasa", "dc73a8ce18e4b8947e8df2ae2958245ca823f3b4", 50.0),
    (
        "diffusion_policy",
        "robocasa-benchmark/diffusion_policy",
        "41212698b6f481ed92a55f0d5f1778ec56bea417",
        20.0,
    ),
    (
        "robomimic",
        "robocasa-benchmark/robomimic",
        "7dc18461c4fc497095f1044f4c40cd44fff62e6a",
        55.0,
    ),
]
BY_NAME = {m[0]: m for m in MANIFEST}


def tarball_url(slug, revision):
    return f"https://codeload.github.com/{slug}/tar.gz/{revision}"


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


def download(name, slug, revision, retries=3, timeout=3600):
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
            print(f"[get ] {name} <- {slug}@{revision[:12]} (第 {attempt} 次)")
            t0 = time.time()
            r = subprocess.run(["curl", "-sL", "--max-time", str(timeout), "-o", str(tgz),
                                "-w", "%{http_code} %{size_download}",
                                tarball_url(slug, revision)], capture_output=True, text=True)
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
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print(f"{'name':<24}{'slug':<48}{'revision':<14}{'MB':>8}")
        for n, s, rev, mb in MANIFEST:
            print(f"{n:<24}{s:<48}{rev[:12]:<14}{mb:>8.1f}")
        return

    DEST.mkdir(parents=True, exist_ok=True)
    requested = set(args.only or ())
    unknown = requested - set(BY_NAME)
    if unknown:
        ap.error(f"unknown repository: {sorted(unknown)}")
    sel = [m for m in MANIFEST if not requested or m[0] in requested]
    print(f"计划拉取 {len(sel)} 个仓库，总计约 {sum(m[3] for m in sel):.1f} MB\n")
    ok = fail = 0
    for n, s, revision, _ in sel:
        if download(n, s, revision):
            ok += 1
        else:
            fail += 1
    print(f"\n完成：成功 {ok}，失败 {fail}")


if __name__ == "__main__":
    main()
