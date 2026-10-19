#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""域标签（style_id / layout_id）读取的测试。

不依赖真实 RoboCasa365 数据：先在本地合成一个"长得一样"的目录树，
覆盖正常与异常情况，确保真实数据到来时不会静默出错。

检查：
  ① build_domain_map 正确扫描 extras → episode_index → (style_id, layout_id)
  ② domain_vocab 生成连续索引
  ③ _attach_domain_fields 产出正确的张量，缺失时给 -1
  ④ 异常情况：缺 ep_meta / 缺 style_id / 目录名不合法 / 根本没有 extras
  ⑤ verify_label_alignment 能发现"标签错位"（最关键的安全网）
  ⑥ 工厂产出的类确实继承官方类
"""
from __future__ import annotations
import json, shutil, sys
from pathlib import Path

import numpy as np
import torch

import dacsc.compat  # noqa
from dacsc.dataset import (build_domain_map, domain_vocab, _attach_domain_fields,
                           verify_label_alignment, build_domain_classes,
                           make_domain_dataset_class)


def make_domain_dataset_class_for_test(base):
    return make_domain_dataset_class(base)

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / ".cache" / "test_domain_labels"


def make_tree(root: Path, episodes, with_meta=True, with_style=True,
              bad_dirname=False, write_episodes_jsonl=True, length_offset=0):
    """合成一个 lerobot 风格目录树。episodes: {episode_index: (style_id, layout_id, length)}"""
    if root.exists():
        shutil.rmtree(root)
    (root / "meta").mkdir(parents=True)
    (root / "extras").mkdir(parents=True)
    eps_lines = []
    for idx, (style, layout, length) in episodes.items():
        name = f"episode_{idx:06d}" if not bad_dirname else f"badname_{idx}"
        d = root / "extras" / name
        d.mkdir(parents=True, exist_ok=True)
        if with_meta:
            meta = {"lang": f"task {idx}", "layout_id": layout}
            if with_style:
                meta["style_id"] = style
            (d / "ep_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        np.savez_compressed(d / "states.npz",
                            states=np.zeros((max(1, length + length_offset), 3)))
        eps_lines.append({"episode_index": idx, "length": length, "tasks": [f"task {idx}"]})
    if write_episodes_jsonl:
        (root / "meta" / "episodes.jsonl").write_text(
            "\n".join(json.dumps(e) for e in eps_lines) + "\n", encoding="utf-8")
    return root


def _test_lazy_module_attr():
    """⑦ 懒加载：hydra 能否通过字符串找到我们的数据集类。

    做法：往 sys.modules 注入一个假的官方模块，模拟"robocasa 已安装"的环境，
    然后验证 `dacsc.dataset.DomainLerobotCotrainingDataset` 能被取到。
    """
    import sys, types
    print("\n⑦ 懒加载机制（hydra 用字符串引用数据集类）")
    fake = types.ModuleType("diffusion_policy.dataset.lerobot_dataset")

    class FakeLerobotDataset:
        def __init__(self, *a, **k): pass

    class FakeLerobotCotrainingDataset:
        def __init__(self, *a, **k): pass

    fake.LerobotDataset = FakeLerobotDataset
    fake.LerobotCotrainingDataset = FakeLerobotCotrainingDataset
    sys.modules["diffusion_policy.dataset.lerobot_dataset"] = fake
    try:
        import importlib
        import dacsc.dataset as dd
        importlib.reload(dd)
        Coto = getattr(dd, "DomainLerobotCotrainingDataset")
        Single = getattr(dd, "DomainLerobotDataset")
        ok = (issubclass(Coto, FakeLerobotCotrainingDataset)
              and issubclass(Single, FakeLerobotDataset))
        # 再验一次 hydra 的字符串解析路径
        import hydra
        cls = hydra.utils.get_class("dacsc.dataset.DomainLerobotCotrainingDataset")
        ok_hydra = (cls is Coto)
        print(f"   dacsc.dataset.DomainLerobotCotrainingDataset 可取到 ? {ok}")
        print(f"   hydra.utils.get_class 能解析同一字符串 ? {ok_hydra}")
        return ok and ok_hydra
    except Exception as e:
        print(f"   ✗ {type(e).__name__}: {e}")
        return False
    finally:
        sys.modules.pop("diffusion_policy.dataset.lerobot_dataset", None)


def main():
    rep = {}
    print("=" * 90)
    EPISODES = {0: (11, 12, 100), 1: (23, 12, 120), 2: (45, 13, 90),
                3: (11, 13, 110), 4: (60, 14, 130)}

    print("① build_domain_map（正常情况）")
    root = make_tree(WORK / "ds_normal", EPISODES)
    m, st = build_domain_map(root)
    print(f"   解析 {st['n_parsed']}/{st['n_episode_dirs']} 个 episode，"
          f"缺 domain 键 {st['n_missing_domain_key']}，坏目录名 {st['n_bad_name']}")
    exp = {i: (v[0], v[1]) for i, v in EPISODES.items()}
    rep["map_ok"] = (m == exp)
    print(f"   映射 == 期望 ? {'✓' if rep['map_ok'] else '✗ ' + str(m)}")

    print("\n② domain_vocab（连续索引）")
    vocab = domain_vocab([m])
    print(f"   style_id → 索引: {vocab}")
    rep["vocab"] = vocab
    print(f"   → {'✓ 连续且保序' if list(vocab.values()) == list(range(len(vocab))) else '✗'}")

    print("\n③ _attach_domain_fields")
    d = _attach_domain_fields({}, m, 2, False, {})
    ok3 = int(d["domain_id"]) == 45 and int(d["layout_id"]) == 13
    d2 = _attach_domain_fields({}, m, 999, False, {})       # 不存在的 episode
    ok3b = int(d2["domain_id"]) == -1 and int(d2["layout_id"]) == -1
    dv = _attach_domain_fields({}, m, 4, True, vocab)       # 用 vocab
    ok3c = int(dv["domain_id"]) == vocab[60]
    rep["attach_ok"] = ok3 and ok3b and ok3c
    print(f"   ep2 → style {int(d['domain_id'])}, layout {int(d['layout_id'])}")
    print(f"   不存在的 ep → -1 ✔? {ok3b}；用 vocab → {int(dv['domain_id'])} (={vocab[60]}) ✔? {ok3c}")
    print(f"   → {'✓' if rep['attach_ok'] else '✗'}")

    print("\n④ 异常情况")
    cases = [
        ("缺 ep_meta.json", dict(with_meta=False), lambda s: s["n_parsed"] == 0),
        ("缺 style_id 键", dict(with_style=False), lambda s: s["n_missing_domain_key"] == 5),
        ("目录名不合法", dict(bad_dirname=True), lambda s: s["n_bad_name"] == 5),
    ]
    ok4 = True
    for name, kw, check in cases:
        r = make_tree(WORK / f"ds_{abs(hash(name))}", EPISODES, **kw)
        mm, ss = build_domain_map(r)
        good = check(ss)
        ok4 &= good
        print(f"   {name:<16} → {('✓' if good else '✗')}  stats={ {k:ss[k] for k in ('n_episode_dirs','n_with_meta','n_parsed','n_missing_domain_key','n_bad_name')} }")
    r = WORK / "ds_no_extras"; (r / "meta").mkdir(parents=True, exist_ok=True)
    mm, ss = build_domain_map(r)
    good = (not ss["exists"]) and len(mm) == 0
    ok4 &= good
    print(f"   {'根本没有 extras':<16} → {('✓' if good else '✗')}  exists={ss['exists']}")
    rep["edge_cases_ok"] = ok4

    print("\n⑤ verify_label_alignment（发现标签错位）")
    r_ok = make_tree(WORK / "ds_align_ok", EPISODES)
    m_ok, _ = build_domain_map(r_ok)
    a_ok = verify_label_alignment(r_ok, m_ok)
    print(f"   对齐时: checked={a_ok['checked']} matched={a_ok['matched']} ok={a_ok.get('ok')}")
    r_bad = make_tree(WORK / "ds_align_bad", EPISODES, length_offset=7)
    m_bad, _ = build_domain_map(r_bad)
    a_bad = verify_label_alignment(r_bad, m_bad)
    print(f"   错位时: checked={a_bad['checked']} 不匹配={len(a_bad['mismatched'])} ok={a_bad.get('ok')}")
    print(f"   示例: {a_bad['mismatched'][:1]}")
    rep["alignment"] = {"ok_case": a_ok.get("ok"), "bad_case_ok": a_bad.get("ok"),
                        "bad_case_n_mismatch": len(a_bad["mismatched"])}
    ok5 = a_ok.get("ok") and (a_bad.get("ok") is False)
    print(f"   → {'✓ 正常能通过、错位能被抓出' if ok5 else '✗ 安全网失效'}")

    print("\n⑥ 工厂类逻辑（用 stub 基类，不依赖真实仿真环境）")
    class StubBase:
        def __init__(self, *a, **k): pass
        def __getitem__(self, i): return {}
    SingleStub = make_domain_dataset_class_for_test(StubBase)
    ok6a = issubclass(SingleStub, StubBase) and hasattr(SingleStub, "__getitem__")
    print(f"   工厂产出类 ⊂ stub 基类 ? {issubclass(SingleStub, StubBase)}")

    print("\n⑥b 真实官方类的继承（需要完整 robocasa 环境）")
    try:
        from diffusion_policy.dataset.lerobot_dataset import (LerobotDataset,
                                                              LerobotCotrainingDataset)
        Single, Coto = build_domain_classes()
        ok6b = issubclass(Single, LerobotDataset) and issubclass(Coto, LerobotCotrainingDataset)
        print(f"   DomainLerobotDataset    ⊂ LerobotDataset ? {issubclass(Single, LerobotDataset)}")
        print(f"   DomainLerobotCotraining ⊂ LerobotCotrainingDataset ? {issubclass(Coto, LerobotCotrainingDataset)}")
        status6b = "PASS" if ok6b else "FAIL"
    except ModuleNotFoundError as e:
        ok6b = True      # 环境问题不算逻辑失败
        status6b = f"SKIP（缺依赖：{e.name}）"
        print(f"   ⊘ 跳过：{e.name} 未安装 —— 训练环境需 pip install robosuite mujoco gymnasium（robocasa 全套）")
    rep["factory_stub_ok"] = ok6a
    rep["factory_official"] = status6b
    rep["factory_ok"] = ok6a and ok6b

    ok7 = _test_lazy_module_attr()
    rep["lazy_load_ok"] = ok7
    rep["all_ok"] = all([rep["map_ok"], rep["attach_ok"], rep["edge_cases_ok"], ok5,
                         rep["factory_ok"], ok7])
    out = ROOT / "audit" / "domain_labels_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n" + "=" * 90)
    print(f"总结：{'✓ 全部通过' if rep['all_ok'] else '✗ 有失败项'}   报告：{out}")
    shutil.rmtree(WORK, ignore_errors=True)
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
