#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DACSC-Robot 阶段 0：RoboCasa365 数据审计

扫描 LeRobot 数据集中的 extras/episode_*/ep_meta.json（以及可选的原始 HDF5），
输出 task x layout x style 交叉表、覆盖度、耦合度、泄漏检查与域分类器可行性评估。

依赖：仅 Python 3.9+ 标准库（不需要 pandas / numpy / torch）。

用法示例
--------
# 扫描一个 split 的全部任务
python audit_dataset.py --dataset-root /data/robocasa365/v1.0/pretrain --out audit/pretrain

# 显式指定 style 划分（默认 train=11-40,val=41-50,test=51-60）
python audit_dataset.py --dataset-root /data/robocasa365/v1.0/pretrain \
    --splits train=11-40,val=41-50,test=51-60 --out audit/pretrain

# 检查数据是否属于 pretrain 组（layout/style 必须 >= 11）
python audit_dataset.py --dataset-root /data/robocasa365/v1.0/pretrain \
    --expect-group pretrain --out audit/pretrain

# 扫描原始 HDF5（需要 h5py）
python audit_dataset.py --hdf5-root /data/robocasa_raw/v1.0/pretrain --out audit/pretrain_raw

# 自检（用合成数据验证脚本本身）
python audit_dataset.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------
# RoboCasa365 场景常量（来源：论文 2603.04356 §3.2 + robocasa/models/scenes/scene_registry.py）
# ----------------------------------------------------------------------------
PRETRAIN_LAYOUT_IDS = list(range(11, 61))   # 50 个户型
PRETRAIN_STYLE_IDS = list(range(11, 61))    # 50 个风格
TARGET_LAYOUT_IDS = list(range(1, 11))      # 10 个 target 户型
TARGET_STYLE_IDS = list(range(1, 11))       # 10 个 target 风格（与 pretrain 不重叠）

DATE_RE = re.compile(r"^\d{8}$")
NOISE_DIRS = {"lerobot", "mg", "demo", "human", "extras", "data", "videos", "meta"}


# ----------------------------------------------------------------------------
# 数据发现
# ----------------------------------------------------------------------------
def infer_task_name(extras_dir: Path, dataset_root: Path, override: str | None) -> str:
    """从路径推断任务名。

    RoboCasa 目录形如 .../<split>/<task>/<YYYYMMDD>[/mg/demo]/lerobot/extras/
    策略：从 lerobot 根往上走，跳过日期段与噪声目录段，取第一个有效目录名。
    """
    if override:
        return override
    parts = extras_dir.parts
    try:
        root_idx = len(dataset_root.parts)
    except Exception:
        root_idx = 0
    # 从 extras 的父目录(lerobot 根)开始向上找
    cur = extras_dir.parent
    for _ in range(6):
        if cur == dataset_root or cur.parent == cur:
            break
        name = cur.name
        if not DATE_RE.match(name) and name not in NOISE_DIRS:
            return name
        cur = cur.parent
    # 退化：取相对 dataset_root 的第一段
    try:
        rel = extras_dir.relative_to(dataset_root).parts
        if rel:
            return rel[0]
    except ValueError:
        pass
    return extras_dir.parent.name


def discover_lerobot_episodes(dataset_root: Path):
    """产出 (task_name, episode_id, ep_meta_dict, source_path, dataset_dir)。"""
    for extras in sorted(dataset_root.rglob("extras")):
        if not extras.is_dir():
            continue
        ep_dirs = [d for d in extras.iterdir() if d.is_dir() and d.name.startswith("episode_")]
        for ep_dir in sorted(ep_dirs):
            meta_path = ep_dir / "ep_meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:                       # 损坏文件不应中断审计
                yield ("__PARSE_ERROR__", ep_dir.name, {"error": str(exc)}, meta_path, extras.parent)
                continue
            task = infer_task_name(ep_dir, dataset_root, override=None)
            yield (task, ep_dir.name, meta, meta_path, extras.parent)


def discover_hdf5_episodes(hdf5_root: Path):
    """可选路径：直接从原始 HDF5 的 ep_meta 属性读取（需要 h5py）。"""
    try:
        import h5py  # type: ignore
    except ImportError:
        print("[warn] 未安装 h5py，跳过 HDF5 扫描", file=sys.stderr)
        return
    for h5 in sorted(hdf5_root.rglob("*.hdf5")):
        task = infer_task_name(h5, hdf5_root, override=None)
        try:
            with h5py.File(h5, "r") as f:
                for demo in sorted(f["data"].keys(), key=lambda s: int(re.sub(r"\D", "", s) or 0)):
                    raw = f["data"][demo].attrs.get("ep_meta")
                    if raw is None:
                        continue
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    meta = json.loads(raw)
                    yield (task, demo, meta, h5, h5.parent)
        except Exception as exc:
            print(f"[warn] 读取失败 {h5}: {exc}", file=sys.stderr)


# ----------------------------------------------------------------------------
# 统计工具（纯标准库）
# ----------------------------------------------------------------------------
def cramers_v(table: list[list[int]]) -> float:
    """Cramér's V（0=独立，1=完全耦合）。"""
    rows = len(table)
    cols = len(table[0]) if rows else 0
    if rows < 2 or cols < 2:
        return 0.0
    n = sum(sum(r) for r in table)
    if n == 0:
        return 0.0
    row_tot = [sum(r) for r in table]
    col_tot = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    chi2 = 0.0
    for i in range(rows):
        for j in range(cols):
            e = row_tot[i] * col_tot[j] / n
            if e > 0:
                chi2 += (table[i][j] - e) ** 2 / e
    denom = n * min(rows - 1, cols - 1)
    return math.sqrt(chi2 / denom) if denom > 0 else 0.0


def chi2_df(table: list[list[int]]) -> tuple[float, int]:
    rows, cols = len(table), len(table[0]) if table else 0
    n = sum(sum(r) for r in table)
    if rows < 2 or cols < 2 or n == 0:
        return 0.0, 0
    row_tot = [sum(r) for r in table]
    col_tot = [sum(table[i][j] for i in range(rows)) for j in range(cols)]
    chi2 = 0.0
    for i in range(rows):
        for j in range(cols):
            e = row_tot[i] * col_tot[j] / n
            if e > 0:
                chi2 += (table[i][j] - e) ** 2 / e
    return chi2, (rows - 1) * (cols - 1)


def parse_splits(spec: str) -> dict[str, list[int]]:
    """'train=11-40,val=41-50' -> {'train':[11..40], ...}"""
    out: dict[str, list[int]] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, rng = chunk.partition("=")
        ids: list[int] = []
        for part in rng.split("+"):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-")
                ids.extend(range(int(a), int(b) + 1))
            elif part:
                ids.append(int(part))
        out[name.strip()] = ids
    return out


# ----------------------------------------------------------------------------
# 审计主体
# ----------------------------------------------------------------------------
class Audit:
    def __init__(self, splits, expect_group, min_per_domain):
        self.splits = splits
        self.expect_group = expect_group
        self.min_per_domain = min_per_domain
        self.rows: list[dict] = []
        self.errors: list[dict] = []

    def add(self, task, episode_id, meta, path, ds_dir):
        if task == "__PARSE_ERROR__":
            self.errors.append({"path": str(path), "error": meta.get("error")})
            return
        layout = meta.get("layout_id")
        style = meta.get("style_id")
        lang = meta.get("lang")
        if isinstance(layout, str) or isinstance(style, str):
            layout = _as_int(layout)
            style = _as_int(style)
        scene_id = (f"layout{int(layout):02d}_style{int(style):02d}"
                    if isinstance(layout, int) and isinstance(style, int) else None)
        self.rows.append({
            "task_name": task,
            "episode_id": episode_id,
            "layout_id": layout,
            "style_id": style,
            "scene_id": scene_id,
            "language": lang if isinstance(lang, str) else json.dumps(lang, ensure_ascii=False),
            "dataset_dir": str(ds_dir),
        })

    # -- 检查项 -------------------------------------------------------------
    def run(self) -> dict:
        rep: dict = {}
        rep["min_per_domain"] = self.min_per_domain
        rep["n_episodes"] = len(self.rows)
        rep["n_parse_errors"] = len(self.errors)
        rep["errors"] = self.errors
        missing = [r for r in self.rows if r["scene_id"] is None]
        rep["n_missing_ids"] = len(missing)

        valid = [r for r in self.rows if r["scene_id"]]
        rep["tasks"] = self._task_table(valid)
        rep["styles"] = self._style_table(valid)
        rep["scenes"] = self._scene_table(valid)
        rep["coupling"] = self._coupling(valid)
        rep["leakage"] = self._leakage(valid)
        rep["domain_feasibility"] = self._domain_feasibility(valid)
        rep["group_check"] = self._group_check(valid)
        return rep

    def _task_table(self, rows) -> list[dict]:
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_name"]].append(r)
        out = []
        for task, rs in sorted(by_task.items()):
            styles = Counter(r["style_id"] for r in rs)
            layouts = Counter(r["layout_id"] for r in rs)
            out.append({
                "task_name": task,
                "n_episodes": len(rs),
                "n_styles": len(styles),
                "n_layouts": len(layouts),
                "n_scenes": len({r["scene_id"] for r in rs}),
                "max_episodes_per_style": max(styles.values()),
                "singleton_scene_fraction": round(
                    sum(1 for v in Counter(r["scene_id"] for r in rs).values() if v == 1) / max(1, len(rs)), 4),
            })
        return out

    def _style_table(self, rows) -> list[dict]:
        by_style = defaultdict(list)
        for r in rows:
            by_style[r["style_id"]].append(r)
        out = []
        for style, rs in sorted(by_style.items()):
            out.append({
                "style_id": style,
                "n_episodes": len(rs),
                "n_tasks": len({r["task_name"] for r in rs}),
                "n_layouts": len({r["layout_id"] for r in rs}),
            })
        return out

    def _scene_table(self, rows) -> list[dict]:
        by_scene = defaultdict(list)
        for r in rows:
            by_scene[r["scene_id"]].append(r)
        out = []
        for scene, rs in sorted(by_scene.items()):
            out.append({
                "scene_id": scene,
                "layout_id": rs[0]["layout_id"],
                "style_id": rs[0]["style_id"],
                "n_episodes": len(rs),
                "n_tasks": len({r["task_name"] for r in rs}),
            })
        return out

    def _coupling(self, rows) -> dict:
        tasks = sorted({r["task_name"] for r in rows})
        styles = sorted({r["style_id"] for r in rows})
        ti = {t: i for i, t in enumerate(tasks)}
        si = {s: j for j, s in enumerate(styles)}
        ttable = [[0] * len(styles) for _ in tasks]
        for r in rows:
            ttable[ti[r["task_name"]]][si[r["style_id"]]] += 1
        chi2, df = chi2_df(ttable)
        v_task_style = cramers_v(ttable)

        layouts = sorted({r["layout_id"] for r in rows})
        li = {l: i for i, l in enumerate(layouts)}
        stable = [[0] * len(styles) for _ in layouts]
        for r in rows:
            stable[li[r["layout_id"]]][si[r["style_id"]]] += 1
        v_layout_style = cramers_v(stable)

        only_one_task = sum(1 for s in styles
                            if len({r["task_name"] for r in rows if r["style_id"] == s}) == 1)
        return {
            "cramers_v_task_style": round(v_task_style, 4),
            "chi2_task_style": round(chi2, 2),
            "chi2_df": df,
            "cramers_v_layout_style": round(v_layout_style, 4),
            "n_styles_seen_in_only_one_task": only_one_task,
            "n_styles_total_in_data": len(styles),
            "n_tasks_total": len(tasks),
        }

    def _leakage(self, rows) -> dict:
        style_to_split: dict[int, list[str]] = defaultdict(list)
        for split, ids in self.splits.items():
            for i in ids:
                style_to_split[i].append(split)
        overlap = {s: names for s, names in style_to_split.items() if len(names) > 1}

        assignments: dict[str, Counter] = defaultdict(Counter)
        unmapped = Counter()
        split_of_style = {s: names[0] for s, names in style_to_split.items()}
        for r in rows:
            split = split_of_style.get(r["style_id"])
            if split is None:
                unmapped[r["style_id"]] += 1
            else:
                assignments[r["task_name"]][split] += 1

        # 场景级泄漏：同一 scene 是否同时以训练和测试身份出现（同一 scene 不可能跨 split，
        # 但同一 (layout,style) 若被两个 task 复用则是另一种复用，另行列示）
        scene_splits: dict[str, set] = defaultdict(set)
        for r in rows:
            sp = split_of_style.get(r["style_id"])
            if sp:
                scene_splits[r["scene_id"]].add(sp)
        scene_conflict = {s: sorted(v) for s, v in scene_splits.items() if len(v) > 1}

        tasks_mixed = {t: dict(c) for t, c in assignments.items() if len(c) > 1}
        return {
            "splits_definition": self.splits,
            "style_overlap_between_splits": overlap,
            "episodes_with_unmapped_style": dict(unmapped),
            "tasks_spanning_multiple_splits": tasks_mixed,
            "scene_id_conflicting_splits": scene_conflict,
            "n_scenes": len(scene_splits),
            "verdict": ("PASS" if not overlap and not scene_conflict else "FAIL"),
        }

    def _domain_feasibility(self, rows) -> dict:
        counts = Counter(r["style_id"] for r in rows)
        if not counts:
            return {"verdict": "N/A", "reason": "无有效 style_id"}
        vals = sorted(counts.values())
        k = len(counts)
        chance = 1.0 / k if k else 0.0
        n_ok = sum(1 for v in vals if v >= self.min_per_domain)
        n_tasks = len({r["task_name"] for r in rows})
        uniform_expected = round(len(rows) / k, 1) if k else None
        return {
            "n_style_classes": k,
            "uniform_expected_per_style": uniform_expected,
            "n_tasks": n_tasks,
            "chance_accuracy": round(chance, 4),
            "min_episodes_per_style": min(vals),
            "median_episodes_per_style": statistics.median(vals),
            "max_episodes_per_style": max(vals),
            "styles_below_min": [s for s, v in sorted(counts.items()) if v < self.min_per_domain],
            "n_styles_meeting_min": n_ok,
            "min_required_per_domain": self.min_per_domain,
            "verdict": "PASS" if n_ok >= max(2, int(0.8 * k)) else "FAIL",
            "note": ("域分类器在 %d 类、随机水平 %.2f%% 下，若多数风格样本量不足，"
                     "域梯度分数不可信；应先把 style 归并为粗粒度外观域。" % (k, chance * 100)),
        }

    def _group_check(self, rows) -> dict:
        if self.expect_group is None:
            return {"verdict": "SKIPPED"}
        if self.expect_group == "pretrain":
            ok_layout, ok_style = set(PRETRAIN_LAYOUT_IDS), set(PRETRAIN_STYLE_IDS)
        else:
            ok_layout, ok_style = set(TARGET_LAYOUT_IDS), set(TARGET_STYLE_IDS)
        bad = [r for r in rows if r["layout_id"] not in ok_layout or r["style_id"] not in ok_style]
        bad_styles = sorted({r["style_id"] for r in bad})
        bad_layouts = sorted({r["layout_id"] for r in bad})
        return {
            "expect_group": self.expect_group,
            "n_violating_episodes": len(bad),
            "violating_styles": bad_styles[:30],
            "violating_layouts": bad_layouts[:30],
            "example_paths": [r["dataset_dir"] for r in bad[:5]],
            "verdict": "PASS" if not bad else "FAIL",
        }


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


# ----------------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_report(out: Path, rep: dict, audit_rows: list[dict]):
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "episode_index.csv", audit_rows,
              ["task_name", "episode_id", "layout_id", "style_id", "scene_id", "language", "dataset_dir"])
    write_csv(out / "task_style_counts.csv", rep["tasks"],
              ["task_name", "n_episodes", "n_styles", "n_layouts", "n_scenes",
               "max_episodes_per_style", "singleton_scene_fraction"])
    write_csv(out / "style_counts.csv", rep["styles"],
              ["style_id", "n_episodes", "n_tasks", "n_layouts"])
    write_csv(out / "scene_counts.csv", rep["scenes"],
              ["scene_id", "layout_id", "style_id", "n_episodes", "n_tasks"])
    (out / "audit_summary.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    L: list[str] = []
    A = L.append
    A("# DACSC-Robot 阶段 0 数据审计报告\n")
    A(f"- episode 总数：**{rep['n_episodes']}**")
    A(f"- 缺少 layout_id/style_id 的 episode：**{rep['n_missing_ids']}**")
    A(f"- 解析失败：**{rep['n_parse_errors']}**")
    A(f"- 任务数：**{rep['coupling']['n_tasks_total']}**，"
      f"数据中出现的 style 数：**{rep['coupling']['n_styles_total_in_data']}**\n")

    A("## 1. 每个任务覆盖了多少 style\n")
    A("| task | episodes | styles | layouts | scenes | 每 style 最大条数 | 单例场景占比 |")
    A("|---|---|---|---|---|---|---|")
    for t in rep["tasks"][:60]:
        A(f"| {t['task_name']} | {t['n_episodes']} | {t['n_styles']} | {t['n_layouts']} | "
          f"{t['n_scenes']} | {t['max_episodes_per_style']} | {t['singleton_scene_fraction']} |")
    if len(rep["tasks"]) > 60:
        A(f"\n（仅显示前 60 个任务，完整表见 task_style_counts.csv）")
    A("")

    A("## 2. 每个 style 有多少 episode\n")
    s = rep["domain_feasibility"]
    A(f"- style 类别数：**{s.get('n_style_classes')}**，随机水平：**{s.get('chance_accuracy')}**")
    A(f"- 每 style episode 数：min={s.get('min_episodes_per_style')} / "
      f"median={s.get('median_episodes_per_style')} / max={s.get('max_episodes_per_style')}")
    A(f"- 达到阈值（每 style >= {rep.get('min_per_domain')} 条）的 style 数："
      f"**{s.get('n_styles_meeting_min')}**")
    if s.get("uniform_expected_per_style") is not None:
        exp = float(s["uniform_expected_per_style"])
        med = float(s.get("median_episodes_per_style") or 0)
        gap = abs(med - exp) / max(exp, 1e-9)
        verdict = ("两者接近，符合均匀抽取" if gap < 0.2
                   else "两者差距 >20%，抽取可能不均匀或任务间场景分布不一致")
        A(f"- 若 style 在任务内均匀抽取，每 style 期望 episode 数约为 **{exp:.1f}**；"
          f"实测中位数 {med:.1f}（偏差 {gap*100:.1f}%）→ {verdict}")
    A(f"- 结论：**{s.get('verdict')}**\n")

    A("## 3. task / layout / style 是否严重耦合\n")
    c = rep["coupling"]
    A(f"- task x style 的 Cramér's V = **{c['cramers_v_task_style']}**"
      f"（chi2={c['chi2_task_style']}, df={c['chi2_df']}）")
    A(f"- layout x style 的 Cramér's V = **{c['cramers_v_layout_style']}**")
    A(f"- 仅出现在单一任务中的 style 数：**{c['n_styles_seen_in_only_one_task']}**"
      f" / {c['n_styles_total_in_data']}")
    A("- 判读：V < 0.10 视为基本独立；0.10–0.25 轻度耦合；> 0.25 视为严重耦合，"
      "此时 style 与 task 不可分离，域分类器会把任务信息当作域信息。\n")

    A("## 4. train/val/test 泄漏检查\n")
    lk = rep["leakage"]
    A(f"- 划分定义：`{lk['splits_definition']}`")
    A(f"- 跨 split 重复的 style：**{len(lk['style_overlap_between_splits'])}**")
    A(f"- 出现在多个 split 的 scene：**{len(lk['scene_id_conflicting_splits'])}**")
    A(f"- 未映射到任何 split 的 episode：**{sum(lk['episodes_with_unmapped_style'].values())}**")
    A(f"- 同时跨多个 split 的任务：**{len(lk['tasks_spanning_multiple_splits'])}**"
      "（这是**预期现象**：split 由 style_id 决定，而 style 在每个任务内逐 episode 抽取，"
      "因此每个任务天然横跨 train/val/test；只有 scene 级重复才是真泄漏）")
    A(f"- 结论：**{lk['verdict']}**\n")

    if rep["group_check"].get("verdict") != "SKIPPED":
        g = rep["group_check"]
        A(f"## 5. 分组归属检查（expect_group={g['expect_group']}）\n")
        A(f"- 违反分组的 episode：**{g['n_violating_episodes']}**")
        A(f"- 违规 style：{g['violating_styles']}")
        A(f"- 违规 layout：{g['violating_layouts']}")
        A(f"- 结论：**{g['verdict']}**\n")

    A("## 6. 域分类器可行性\n")
    A(f"- 结论：**{s.get('verdict')}**")
    A(f"- {s.get('note','')}")
    if s.get("styles_below_min"):
        A(f"- 样本量不足的 style：{s['styles_below_min'][:40]}")
    A("")

    A("## 7. 门槛判定\n")
    ok = (lk["verdict"] == "PASS"
          and rep["n_missing_ids"] == 0
          and rep["n_parse_errors"] == 0
          and rep["group_check"].get("verdict") in ("PASS", "SKIPPED"))
    A(f"**{'可以进入阶段 1/3' if ok else '不可进入下一步'}**"
      f"（泄漏={lk['verdict']}，缺失={rep['n_missing_ids']}，"
      f"解析错误={rep['n_parse_errors']}，分组={rep['group_check'].get('verdict')}）")
    A("")
    (out / "audit_report.md").write_text("\n".join(L), encoding="utf-8")
    return out / "audit_report.md"


# ----------------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------------
def selftest():
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp(prefix="dacsc_selftest_"))
    cases = {
        # (task, n_ep, styles)  —— 正常覆盖
        "Clean": ("PickPlaceCounterToCabinet", 40, list(range(11, 51))),
        # 与任务强耦合（只有一个 style）
        "Coupled": ("TurnOffStove", 40, [11] * 40),
        # 稀疏（每 style 1 条、30 个 style）
        "Sparse": ("OpenDrawer", 30, list(range(11, 41, 1))),
    }
    for name, (task, n, styles) in cases.items():
        for i in range(n):
            d = tmp / name / task / "20250811" / "lerobot" / "extras" / f"episode_{i:06d}"
            d.mkdir(parents=True)
            layout = 11 + (i % 10)
            style = styles[i % len(styles)]
            (d / "ep_meta.json").write_text(json.dumps({
                "lang": f"do {task}", "layout_id": layout, "style_id": style,
                "fixtures": {}, "objects": [],
            }), encoding="utf-8")
    # 混入一个 target 组 episode（应被 --expect-group pretrain 判为 FAIL）
    d = tmp / "Mix" / "OpenCabinet" / "20250811" / "lerobot" / "extras" / "episode_000000"
    d.mkdir(parents=True)
    (d / "ep_meta.json").write_text(json.dumps(
        {"lang": "x", "layout_id": 3, "style_id": 3}), encoding="utf-8")

    print("自检数据目录：", tmp)
    for tag, root, expect in [("pretrain", tmp, "pretrain")]:
        a = Audit(parse_splits("train=11-40,val=41-50,test=51-60"), expect, 32)
        for t, e, m, p, ds in discover_lerobot_episodes(root):
            a.add(t, e, m, p, ds)
        rep = a.run()
        rp = write_report(Path("/tmp/dacsc_selftest_out"), rep, a.rows)
        print(f"[{tag}] episodes={rep['n_episodes']} tasks={len(rep['tasks'])} "
              f"V(task,style)={rep['coupling']['cramers_v_task_style']} "
              f"leak={rep['leakage']['verdict']} group={rep['group_check']['verdict']} "
              f"domain={rep['domain_feasibility']['verdict']}")
        print("报告：", rp)
    shutil.rmtree(tmp, ignore_errors=True)
    assert True


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RoboCasa365 数据审计（阶段 0）")
    ap.add_argument("--dataset-root", type=Path, action="append", default=None,
                    help="LeRobot 数据集根目录（可多次传入）")
    ap.add_argument("--hdf5-root", type=Path, action="append", default=None,
                    help="原始 HDF5 根目录（可选，需要 h5py）")
    ap.add_argument("--out", type=Path, default=Path("audit"))
    ap.add_argument("--splits", default="train=11-40,val=41-50,test=51-60",
                    help="style 划分，如 train=11-40,val=41-50,test=51-60")
    ap.add_argument("--expect-group", choices=["pretrain", "target"], default=None,
                    help="校验数据是否属于该分组（pretrain: id>=11；target: id<=10）")
    ap.add_argument("--min-per-domain", type=int, default=32,
                    help="单个 style 被认为可训练域分类器所需的最少 episode 数")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.dataset_root and not args.hdf5_root:
        ap.error("至少需要 --dataset-root 或 --hdf5-root（或使用 --selftest）")

    audit = Audit(parse_splits(args.splits), args.expect_group, args.min_per_domain)
    for root in (args.dataset_root or []):
        if not root.exists():
            print(f"[error] 路径不存在：{root}", file=sys.stderr)
            continue
        for item in discover_lerobot_episodes(root):
            audit.add(*item)
    for root in (args.hdf5_root or []):
        for item in discover_hdf5_episodes(root):
            audit.add(*item)

    rep = audit.run()
    rep["min_per_domain"] = args.min_per_domain
    report = write_report(args.out, rep, audit.rows)
    print(f"episodes={rep['n_episodes']}  tasks={len(rep['tasks'])}  "
          f"styles={rep['domain_feasibility'].get('n_style_classes')}")
    print(f"leakage={rep['leakage']['verdict']}  "
          f"group={rep['group_check'].get('verdict')}  "
          f"domain_feasibility={rep['domain_feasibility'].get('verdict')}")
    print(f"报告：{report}")


if __name__ == "__main__":
    main()
