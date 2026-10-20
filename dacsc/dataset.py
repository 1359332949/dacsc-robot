"""给官方 LeRobot 数据集补上**域标签**（RoboCasa365 的 style_id / layout_id）。

为什么需要：
    我们的方法（通道级域敏感性）完全依赖外观域标签。官方 `LerobotDataset` 只提供
    图像 / 状态 / 动作 / 语言嵌入，**不含 style_id**。所以必须自己补。

标签从哪来：
    RoboCasa365 的每个 episode 都落盘一个
        <lerobot 根>/extras/episode_XXXXXX/ep_meta.json
    其中含 `layout_id`、`style_id`、`gen_textures`、`lang`。参见
    repos/robocasa/robocasa/environments/kitchen/kitchen.py 的 get_ep_meta()。

怎么接：
    官方 `LerobotDataset.__getitem__` 里已经用 `self.all_steps[idx] → (trajectory_id, step)`
    查语言嵌入表；域标签**照同一个模式**做一张 episode_index → (style_id, layout_id) 的表。

设计原则：**不改动官方文件**，全部通过继承实现。
"""
from __future__ import annotations

import functools
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Subset

log = logging.getLogger(__name__)

EXTRAS_DIRNAME = "extras"
_EPISODE_DIR_RE = re.compile(r"episode_(\d+)")
EP_META_FILENAME = "ep_meta.json"
DEFAULT_DOMAIN_KEY = "style_id"      # 外观域标签
DEFAULT_LAYOUT_KEY = "layout_id"     # 几何标签（控制变量）


# ----------------------------------------------------------------------------
def build_domain_map(dataset_path: str | Path,
                     domain_key: str = DEFAULT_DOMAIN_KEY,
                     layout_key: str = DEFAULT_LAYOUT_KEY,
                     ) -> Tuple[Dict[int, Tuple[Optional[int], Optional[int]]], Dict[str, int]]:
    """扫描 extras/ 建立 `episode_index → (domain_value, layout_value)` 映射。

    返回 (映射表, 统计信息)。映射表的 key 是 episode 序号（与 LeRobot 的 episode_index 对齐）。
    """
    root = Path(dataset_path)
    extras = root / EXTRAS_DIRNAME
    stats = {"extras_dir": str(extras), "exists": extras.is_dir(),
             "n_episode_dirs": 0, "n_with_meta": 0, "n_parsed": 0,
             "n_missing_domain_key": 0, "n_bad_name": 0}
    mapping: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
    if not extras.is_dir():
        log.warning("找不到 %s —— 该数据集将没有域标签（domain_id 恒为 -1）", extras)
        return mapping, stats

    for ep_dir in sorted(extras.iterdir()):
        if not ep_dir.is_dir():
            continue
        stats["n_episode_dirs"] += 1
        meta_f = ep_dir / EP_META_FILENAME
        if not meta_f.is_file():
            continue
        stats["n_with_meta"] += 1
        m = _EPISODE_DIR_RE.fullmatch(ep_dir.name)   # 严格匹配，避免误收非 episode 目录
        if m is None:
            stats["n_bad_name"] += 1
            continue
        idx = int(m.group(1))
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
        except Exception as exc:                     # 损坏文件不中断
            log.warning("解析失败 %s: %s", meta_f, exc)
            continue
        if domain_key not in meta:
            stats["n_missing_domain_key"] += 1
        dv = meta.get(domain_key)
        lv = meta.get(layout_key)
        mapping[idx] = (int(dv) if dv is not None else None,
                        int(lv) if lv is not None else None)
        stats["n_parsed"] += 1
    return mapping, stats


def domain_vocab(mappings: List[Dict[int, Tuple[Optional[int], Optional[int]]]],
                 which: int = 0) -> Dict[int, int]:
    """把多个数据集的域取值汇总成一个连续索引映射：原始 style_id → 0..N-1。

    which=0 取域标签（style_id），which=1 取布局标签（layout_id）。
    """
    vals = set()
    for m in mappings:
        for tup in m.values():
            v = tup[which]
            if v is not None:
                vals.add(int(v))
    return {v: i for i, v in enumerate(sorted(vals))}


def verify_label_alignment(dataset_path: str | Path,
                           mapping: Dict[int, Tuple[Optional[int], Optional[int]]],
                           n_check: int = 20) -> dict:
    """校验「extras 序号」确实等于「LeRobot episode_index」。

    这是**最容易错且错得最隐蔽**的地方：如果两者错位，域标签会整体张冠李戴，
    而所有下游代码都不会报错。

    做法：LeRobot 的 meta/episodes.jsonl 里每个 episode 有 `length`（帧数）；
    同序号 extras 目录下的 states.npz 里 `states` 的行数应当与之相等。
    """
    root = Path(dataset_path)
    res = {"checked": 0, "matched": 0, "mismatched": [], "skipped": None}
    eps_jsonl = root / "meta" / "episodes.jsonl"
    if not eps_jsonl.is_file():
        res["skipped"] = f"找不到 {eps_jsonl}"
        return res
    try:
        eps = [json.loads(l) for l in eps_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception as exc:
        res["skipped"] = f"解析 episodes.jsonl 失败: {exc}"
        return res
    lens = {e.get("episode_index"): e.get("length") for e in eps}
    import numpy as np
    for idx, _ in sorted(mapping.items())[:n_check]:
        n_expected = lens.get(idx)
        npz = root / EXTRAS_DIRNAME / f"episode_{idx:06d}" / "states.npz"
        if n_expected is None or not npz.is_file():
            continue
        try:
            with np.load(npz) as z:
                n_actual = int(z["states"].shape[0])
        except Exception:
            continue
        res["checked"] += 1
        if n_actual == int(n_expected):
            res["matched"] += 1
        else:
            res["mismatched"].append({"episode_index": idx,
                                      "episodes_jsonl_length": int(n_expected),
                                      "states_npz_rows": n_actual})
    res["ok"] = (res["checked"] > 0 and not res["mismatched"])
    return res


# ----------------------------------------------------------------------------
def _attach_domain_fields(data: dict, mapping, trajectory_id, use_vocab, vocab) -> dict:
    tup = mapping.get(int(trajectory_id))
    if tup is None:
        data["domain_id"] = torch.tensor(-1, dtype=torch.long)
        data["layout_id"] = torch.tensor(-1, dtype=torch.long)
        return data
    dom, lay = tup
    if dom is None:
        data["domain_id"] = torch.tensor(-1, dtype=torch.long)
    elif use_vocab:
        data["domain_id"] = torch.tensor(vocab.get(int(dom), -1), dtype=torch.long)
    else:
        data["domain_id"] = torch.tensor(int(dom), dtype=torch.long)
    data["layout_id"] = torch.tensor(int(lay) if lay is not None else -1, dtype=torch.long)
    return data


# ----------------------------------------------------------------------------
def make_domain_dataset_class(LerobotDataset,                 # 官方类
                              domain_key: str = DEFAULT_DOMAIN_KEY,
                              layout_key: str = DEFAULT_LAYOUT_KEY):
    """工厂：生成一个继承官方 `LerobotDataset` 且带域标签的类。

    这样即使官方类被替换/更新，我们也不需要硬编码导入路径。
    """

    class _DomainLerobotDataset(LerobotDataset):
        """官方 LerobotDataset + `domain_id` / `layout_id` 字段。"""

        def __init__(self, *args, use_domain_vocab: bool = False,
                     domain_vocab_map: Optional[Dict[int, int]] = None,
                     verify_alignment: bool = True, **kwargs):
            super().__init__(*args, **kwargs)
            self.use_domain_vocab = use_domain_vocab
            self._domain_vocab_map = domain_vocab_map or {}
            self.domain_map, self.domain_stats = build_domain_map(
                self.dataset_path, domain_key=domain_key, layout_key=layout_key)
            self.domain_stats["dataset_path"] = str(self.dataset_path)
            self.domain_stats["n_trajectories"] = int(len(self.trajectory_ids))
            # 一致性检查：extras 数量应与 episode 数一致（filter_key 子集除外）
            self.domain_stats["covers_all_episodes"] = (
                len(self.domain_map) >= len(self.trajectory_ids))
            self._alignment = None
            if verify_alignment:
                self.domain_stats["alignment"] = self.alignment_report()
            if not self.domain_stats["exists"]:
                log.warning("数据集 %s 没有 extras/ → 无域标签", self.dataset_path)
            elif not self.domain_stats["covers_all_episodes"]:
                log.warning("数据集 %s：extras 只有 %d 条，但 episode 有 %d 条，"
                            "可能有域标签缺失", self.dataset_path,
                            len(self.domain_map), len(self.trajectory_ids))

        def alignment_report(self) -> dict:
            """标签对齐校验结果（首次调用时计算并缓存）。"""
            if self._alignment is None:
                self._alignment = verify_label_alignment(self.dataset_path, self.domain_map)
                if self._alignment.get("checked", 0) > 0 and not self._alignment.get("ok", True):
                    log.error("⚠ 域标签可能错位！%s", self._alignment["mismatched"][:3])
            return self._alignment

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            data = super().__getitem__(idx)
            trajectory_id, _ = self.all_steps[idx]          # 与官方查语言嵌入同一模式
            return _attach_domain_fields(data, self.domain_map, trajectory_id,
                                         self.use_domain_vocab, self._domain_vocab_map)

    _DomainLerobotDataset.__name__ = "DomainLerobotDataset"
    return _DomainLerobotDataset


def make_domain_cotraining_class(LerobotCotrainingDataset,   # 官方多数据集类
                                 domain_key: str = DEFAULT_DOMAIN_KEY,
                                 layout_key: str = DEFAULT_LAYOUT_KEY):
    """工厂：生成继承官方 `LerobotCotrainingDataset` 且子数据集带域标签的类。

    官方该类的 __init__ 在列表推导里**硬编码** `LerobotDataset(...)`，
    为不改官方文件，这里在调用前**临时替换模块级符号**（作用域最小、有 finally 复位）。
    """

    class _DomainLerobotCotrainingDataset(LerobotCotrainingDataset):
        def __init__(self, *args, **kwargs):
            self._domain_balance = bool(kwargs.pop("domain_balance", False))
            self._domain_batch_size = int(kwargs.pop("domain_batch_size", 32))
            self._domain_id_min = int(kwargs.pop("domain_id_min", 11))
            self._domain_group_size = int(kwargs.pop("domain_group_size", 5))
            self._domain_samples_per_group = int(
                kwargs.pop("domain_samples_per_group", 3))
            self._domain_max_groups_per_batch = kwargs.pop(
                "domain_max_groups_per_batch", None)
            self._val_style_min = kwargs.pop("val_style_min", None)
            self._val_style_max = kwargs.pop("val_style_max", None)
            self._val_max_steps = int(kwargs.pop("val_max_steps", 2048))
            import diffusion_policy.dataset.lerobot_dataset as _mod
            orig = _mod.LerobotDataset
            _mod.LerobotDataset = functools.partial(
                make_domain_dataset_class(orig, domain_key=domain_key, layout_key=layout_key))
            try:
                super().__init__(*args, **kwargs)
            finally:
                _mod.LerobotDataset = orig
            self._domain_seed = int(getattr(self, "seed", 42))
            # 汇总各子数据集的统计，便于日志检查
            self.domain_stats_all = [getattr(d, "domain_stats", {}) for d in self.datasets]
            self._val_indices = self._build_style_val_indices()

        def _build_style_val_indices(self):
            """从 domain sampler 缓存中取 style 41–50 的索引，作为留出验证集。"""
            if self._val_style_min is None or self._val_style_max is None:
                return []
            lo, hi = int(self._val_style_min), int(self._val_style_max)
            groups = sorted({
                max(0, (s - self._domain_id_min) // self._domain_group_size)
                for s in range(lo, hi + 1)
            })
            from dacsc.sampler import DomainBalancedSampler
            smp = DomainBalancedSampler(
                self, batch_size=self._domain_batch_size,
                domain_id_min=self._domain_id_min,
                domain_group_size=self._domain_group_size,
                samples_per_domain=self._domain_samples_per_group,
                max_groups_per_batch=self._domain_max_groups_per_batch,
                seed=self._domain_seed)
            smp._build()
            idx = []
            for g in groups:
                idx.extend(smp._groups.get(g, []))
            idx = sorted(set(int(i) for i in idx))
            if len(idx) > self._val_max_steps:
                step = max(1, len(idx) // self._val_max_steps)
                idx = idx[::step][:self._val_max_steps]
            print(f"[DomainLerobotCotrainingDataset] val styles {lo}-{hi}: "
                  f"{len(idx)} steps, groups={groups}", flush=True)
            return idx

        def get_validation_dataset(self):
            if not self._val_indices:
                return Subset(self, [])
            return Subset(self, self._val_indices)

        def get_dataset_sampler(self):
            if not self._domain_balance:
                return None
            from dacsc.sampler import DomainBalancedSampler
            return DomainBalancedSampler(
                self, batch_size=self._domain_batch_size,
                domain_id_min=self._domain_id_min,
                domain_group_size=self._domain_group_size,
                samples_per_domain=self._domain_samples_per_group,
                max_groups_per_batch=self._domain_max_groups_per_batch,
                seed=self._domain_seed)

    _DomainLerobotCotrainingDataset.__name__ = "DomainLerobotCotrainingDataset"
    return _DomainLerobotCotrainingDataset


# ----------------------------------------------------------------------------
def build_domain_classes(domain_key: str = DEFAULT_DOMAIN_KEY,
                         layout_key: str = DEFAULT_LAYOUT_KEY):
    """便捷入口：返回 (单数据集类, 多数据集类)。"""
    from diffusion_policy.dataset.lerobot_dataset import (
        LerobotDataset, LerobotCotrainingDataset)
    return (make_domain_dataset_class(LerobotDataset, domain_key, layout_key),
            make_domain_cotraining_class(LerobotCotrainingDataset, domain_key, layout_key))


# ----------------------------------------------------------------------------
# PEP 562 懒加载：让 hydra 能写 `_target_: dacsc.dataset.DomainLerobotCotrainingDataset`
# 而本地 `import dacsc.dataset` 仍然不需要 robocasa（只在真正实例化时才构造类）。
# ----------------------------------------------------------------------------
_LAZY_CACHE: dict = {}


def __getattr__(name):                      # noqa: N807
    """懒加载 + **缓存**。

    必须缓存：否则每次属性访问都会新建一个类，
    导致 `isinstance` 失效、hydra 实例化得到与检查时不同的类。
    """
    if name in ("DomainLerobotDataset", "DomainLerobotCotrainingDataset"):
        if name not in _LAZY_CACHE:
            Single, Coto = build_domain_classes()
            _LAZY_CACHE["DomainLerobotDataset"] = Single
            _LAZY_CACHE["DomainLerobotCotrainingDataset"] = Coto
        return _LAZY_CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
