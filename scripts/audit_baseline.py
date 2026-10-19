#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DACSC-Robot 阶段 1：官方 Diffusion Policy 基线静态审计

在正式训练前，把以下事实固定下来并留档：
  1. 解析官方 hydra 配置，得到真实的键名与取值（horizon / n_obs_steps / n_action_steps /
     crop_shape / batch_size / obs 键与 shape / action dim / DiT 维度）
  2. 验证 ResNet18ConvFiLM 的 block 拆分与每层通道数（torch 可用时真跑一遍）
  3. 读取每个任务的 horizon（robocasa/utils/dataset_registry.py）
  4. 记录显存需求：官方建议值 + （有 CUDA 时）实测峰值

依赖：Python 3.9+ 标准库。torch / torchvision 为可选项（缺失时退化为静态分析并明确标注）。

用法
----
python audit_baseline.py \
    --dp-repo        /path/to/diffusion_policy \
    --robomimic-repo /path/to/robomimic \
    --robocasa-repo  /path/to/robocasa \
    --out audit/baseline

# 只看配置与 horizon（不需要 torch）
python audit_baseline.py --dp-repo ... --robocasa-repo ... --no-torch
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

# 官方文档给出的硬件建议（robocasa.ai/docs .../policy_learning_algorithms.html）
OFFICIAL_SPECS = {
    "train_gpu_min_gb": 24,
    "train_gpu_recommended_gb": 48,
    "inference_gpu_min_gb": 8,
    "eval_instance": "16 core / 64 GB RAM（env_runner 注释）",
    "source": "robocasa.ai 官方文档 + diffusion_policy README",
}


# ---------------------------------------------------------------------------
# 1) 极简 YAML 解析（只处理本配置用到的结构，避免引入 PyYAML 之外的重依赖）
# ---------------------------------------------------------------------------
def _strip_comment(line: str) -> str:
    out = []
    in_s = in_d = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            if i == 0 or line[i - 1] in " \t":
                break
        out.append(ch)
    return "".join(out).rstrip()


def parse_simple_yaml(path: Path) -> dict:
    """把扁平/一层嵌套的 YAML 解析成 dict（键 -> 原始字符串值或 dict）。"""
    root: dict = {}
    anchors: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        if body.startswith("- "):
            key = parent.setdefault("__list__", [])
            key.append(body[2:].strip())
            continue
        if ":" not in body:
            continue
        key, _, val = body.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "" or re.fullmatch(r"&\w+", val):
            if re.fullmatch(r"&\w+", val):
                anchors[val[1:]] = key
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            if val.startswith("*"):
                ref = anchors.get(val[1:])
                parent[key] = parent.get(ref, val) if ref else val
            else:
                parent[key] = val
    return root


def dig(cfg: dict, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolve(value, cfg: dict, depth: int = 0):
    """解析 ${a.b} / ${eval:'...'} 这类引用（仅覆盖本配置用到的形式）。"""
    if not isinstance(value, str) or depth > 6:
        return value
    m = re.fullmatch(r"\$\{([^}]+)\}", value.strip())
    if not m:
        return re.sub(r"\$\{([^}]+)\}",
                      lambda mm: str(resolve("${%s}" % mm.group(1), cfg, depth + 1)), value)
    expr = m.group(1).strip()
    ev = re.fullmatch(r"eval:'(.+)'", expr)
    if ev:
        inner = ev.group(1)
        # 把 ${x} 替换为数值后做一次安全的算术求值
        def sub(mm):
            v = resolve("${%s}" % mm.group(1), cfg, depth + 1)
            return str(v)
        inner = re.sub(r"\$\{([^}]+)\}", sub, inner)
        if re.fullmatch(r"[-+*/(). 0-9]+", inner):
            try:
                return eval(inner, {"__builtins__": {}}, {})   # 仅数值算术
            except Exception:
                return inner
        return inner
    return resolve(dig(cfg, expr, value), cfg, depth + 1)


# ---------------------------------------------------------------------------
# 2) ResNet18ConvFiLM 的 block 结构
# ---------------------------------------------------------------------------
def analyze_blocks_native(robomimic_repo: Path | None, sizes=(224, 256)):
    """真跑一遍：torch 可用时直接实例化并 hook 每个 _conv_blocks[i] 的输出。"""
    try:
        import torch
        import torchvision
        import numpy as np
    except Exception as exc:
        return {"method": "unavailable", "reason": f"torch/torchvision 不可用：{exc}"}

    import torch.nn as nn

    cls_name = "ResNet18ConvFiLM"
    src_path = None
    if robomimic_repo:
        cand = robomimic_repo / "robomimic" / "models" / "base_nets.py"
        if cand.is_file():
            src_path = cand
    if src_path:
        try:
            sys.path.insert(0, str(robomimic_repo))
            from robomimic.models.base_nets import ResNet18ConvFiLM as _Cls  # type: ignore
            model = _Cls(lang_emb_dim=768, input_channel=3)
            method = "native_import"
        except Exception as exc:
            model = _replicated(robomimic_repo, nn, torchvision)
            method = f"replicated_logic（native import 失败：{type(exc).__name__}）"
    else:
        model = _replicated(robomimic_repo, nn, torchvision)
        method = "replicated_logic（未提供 --robomimic-repo）"

    model.eval()
    report = {"method": method, "sizes": {}}
    with torch.no_grad():
        for size in sizes:
            x = torch.zeros(1, 3, size, size)
            lang = torch.zeros(1, 768)
            feats = []
            handles = [blk.register_forward_hook(
                lambda m, i, o, _f=feats: _f.append(tuple(o.shape))) for blk in model._conv_blocks]
            base = model._base_block(x)
            y = base
            for conv, film in zip(model._conv_blocks, model._film_layers):
                y = conv(y)
                y = film(y, lang)
            for h in handles:
                h.remove()
            report["sizes"][size] = {
                "base_block_out": list(base.shape[1:]),
                "conv_block_out": [list(s[1:]) for s in feats],
                "final_out": list(y.shape[1:]),
            }
    report["n_conv_blocks"] = len(model._conv_blocks)
    report["channels"] = [s[1] for s in feats]
    report["film_layers"] = len(model._film_layers)
    report["film_lang_dim"] = 768
    return report


def _replicated(robomimic_repo, nn, torchvision):
    """复刻 fork 源码 (base_nets.py) 中 ResNet18ConvFiLM 的拆分逻辑。"""
    class FiLM(nn.Module):
        def __init__(self, lang_dim, channels):
            super().__init__()
            self.lang_proj = nn.Linear(lang_dim, channels * 2)
            self.relu = nn.ReLU()

        def forward(self, x, lang):
            b, c, h, w = x.shape
            beta, gamma = torch.split(self.lang_proj(lang).reshape(b, c * 2, 1, 1), [c, c], 1)
            return self.relu((1 + gamma) * x + beta)

    import torch

    class ResNet18ConvFiLMReplica(nn.Module):
        def __init__(self, lang_emb_dim=768, input_channel=3):
            super().__init__()
            net = torchvision.models.resnet18(weights=None)
            layers = nn.ModuleList(net.children())
            base_block, conv_blocks = [], []
            for layer in layers:
                if isinstance(layer, nn.Sequential):
                    for sub in layer:
                        conv_blocks.append(sub)
                elif len(conv_blocks) == 0:
                    base_block.append(layer)
            self._base_block = nn.Sequential(*base_block)
            self._conv_blocks = nn.ModuleList(conv_blocks)
            film_layers, cur = [], self._base_block(torch.rand((1, input_channel, 3, 3))).shape[1]
            for conv in conv_blocks:
                cur = conv(torch.rand((1, cur, 3, 3))).shape[1]
                film_layers.append(FiLM(lang_emb_dim, cur))
            self._film_layers = nn.ModuleList(film_layers)
            self._output_channels = cur

    return ResNet18ConvFiLMReplica()


def analyze_blocks_static(robomimic_repo: Path | None, sizes=(224, 256)):
    """不依赖 torch：从源码推导 block 数与通道数，并用源码自带的 ceil(H/32) 规则算空间尺寸。"""
    chans = [64, 64, 128, 128, 256, 256, 512, 512]
    strides = [4, 4, 8, 8, 16, 16, 32, 32]
    out = {"method": "static_source_analysis", "n_conv_blocks": len(chans),
           "channels": chans, "film_layers": len(chans), "film_lang_dim": 768,
           "sizes": {}, "evidence": []}
    src = None
    if robomimic_repo:
        cand = robomimic_repo / "robomimic" / "models" / "base_nets.py"
        if cand.is_file():
            src = cand.read_text(encoding="utf-8")
    if src:
        for pat in [r"for layer in layers:\s*\n\s*if isinstance\(layer, nn\.Sequential\)",
                    r"film_layers\.append\(FiLMLayer\(lang_emb_dim, current_channels\)\)",
                    r"out_h = int\(math\.ceil\(input_shape\[1\] / 32\.\)\)"]:
            out["evidence"].append({"pattern": pat, "found": bool(re.search(pat, src))})
    else:
        out["evidence"].append({"note": "未提供 --robomimic-repo，通道数来自源码阅读结论，未被机器校验"})
    for size in sizes:
        dims = []
        for ch, st in zip(chans, strides):
            f = math.ceil(size / st)
            dims.append([ch, f, f])
        base = math.ceil(size / 4)   # conv1(stride2) + maxpool(stride2)
        out["sizes"][size] = {"base_block_out": [64, base, base],
                              "conv_block_out": dims,
                              "final_out": dims[-1],
                              "note": "空间尺寸按 fork 的 output_shape() 规则 ceil(H/32) 推导；"
                                      "base_block = conv1+bn+relu+maxpool（stride 4）"}
    return out


# ---------------------------------------------------------------------------
# 3) 每任务 horizon
# ---------------------------------------------------------------------------
def parse_task_horizons(robocasa_repo: Path | None):
    if not robocasa_repo:
        return {"status": "skipped", "reason": "未提供 --robocasa-repo"}
    path = robocasa_repo / "robocasa" / "utils" / "dataset_registry.py"
    if not path.is_file():
        return {"status": "skipped", "reason": f"找不到 {path}"}
    src = path.read_text(encoding="utf-8")
    horizons: dict[str, int] = {}
    cur_task = None
    for line in src.splitlines():
        m = re.match(r"^\s{4}([A-Za-z][A-Za-z0-9_]*)=dict\(", line)
        if m:
            cur_task = m.group(1)
            continue
        m = re.match(r"^\s+horizon=(\d+),", line)
        if m and cur_task:
            horizons.setdefault(cur_task, int(m.group(1)))
            continue
        if re.match(r"^[A-Z_]+ = ", line):
            cur_task = None
    vals = sorted(horizons.values())
    out = {"status": "ok", "n_tasks": len(horizons),
           "min": vals[0] if vals else None, "max": vals[-1] if vals else None,
           "is_multiple_of_150": all(v % 150 == 0 for v in vals) if vals else None,
           "note": "v1.0.1 起所有任务 horizon ×1.5，因此均为 150 的倍数",
           "distribution": {}}
    for v in vals:
        out["distribution"][str(v)] = out["distribution"].get(str(v), 0) + 1
    # 关注官方评测任务集的 horizon
    try:
        tree = ast.parse(src)
        registries = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        registries[t.id] = node
        def eval_str(node):
            try:
                return ast.literal_eval(node)
            except Exception:
                return None
        sets = {}
        for name in ("TASK_SET_REGISTRY", "DATASET_SOUP_REGISTRY"):
            node = registries.get(name)
            if node is None:
                continue
            for k, v in zip(node.value.keys, node.value.values):
                key = eval_str(k)
                if key in ("atomic_seen", "target50", "pretrain300", "all_tasks"):
                    task_list = [x for x in re.findall(r"\"([A-Za-z0-9_]+)\"", ast.dump(v))]
                    sets[f"{name}.{key}"] = len(task_list)
        out["registry_entries"] = sets
    except Exception as exc:
        out["registry_parse_error"] = str(exc)
    out["per_task"] = dict(sorted(horizons.items()))
    return out


# ---------------------------------------------------------------------------
# 4) 显存
# ---------------------------------------------------------------------------
def estimate_memory(dp_repo: Path | None, batch_size, cfg):
    res = {"official": OFFICIAL_SPECS}
    if dp_repo:
        readme = dp_repo / "README.md"
        if readme.is_file():
            txt = readme.read_text(encoding="utf-8")
            m = re.search(r"at least (\d+) Gb.*?([\d]+) Gb\+", txt, re.I | re.S)
            res["readme_says"] = m.group(0).replace("\n", " ") if m else None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(batch_size, 3, 224, 224, device="cuda")
            net = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.ReLU(),
                torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
            ).cuda()
            opt = torch.optim.Adam(net.parameters())
            loss = net(x).mean(); loss.backward(); opt.step()
            res["measured_peak_gb_probe"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            res["device"] = torch.cuda.get_device_name(0)
            res["measured_note"] = ("这是探针网络的峰值，仅用于确认设备可用；"
                                    "真实 DP 训练峰值须用 --config-name=train_diffusion_transformer_bs192 跑 N 步实测。")
        else:
            res["measured"] = "CUDA 不可用，未实测"
    except Exception as exc:
        res["measured"] = f"torch 不可用：{exc}"
    res["analytic_note"] = ("DP(DiT-512, bs192, 3×224×224 输入) 的显存主要由三段决定："
                            "①视觉编码器激活（3 路相机 × ResNet18，crop 224）；"
                            "②DiT 12 层/512 维、horizon=10 的注意力激活；"
                            "③EMA 影子参数 + Adam 状态（参数量的 3 倍）。"
                            "官方建议 ≥24 GB、48 GB+ 更佳，可作为硬约束。")
    return res


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="官方 DP 基线静态审计（阶段 1）")
    ap.add_argument("--dp-repo", type=Path, required=True)
    ap.add_argument("--robomimic-repo", type=Path, default=None)
    ap.add_argument("--robocasa-repo", type=Path, default=None)
    ap.add_argument("--config-name", default="train_diffusion_transformer_bs192")
    ap.add_argument("--task-config", default="robocasa/pretrain_human300")
    ap.add_argument("--out", type=Path, default=Path("audit/baseline"))
    ap.add_argument("--no-torch", action="store_true")
    args = ap.parse_args()

    rep: dict = {"config_name": args.config_name, "task_config": args.task_config}

    # 1) 配置
    train_path = args.dp_repo / "diffusion_policy" / "config" / f"{args.config_name}.yaml"
    task_path = args.dp_repo / "diffusion_policy" / "config" / "task" / f"{args.task_config}.yaml"
    rep["paths"] = {"train": str(train_path), "task": str(task_path),
                    "train_exists": train_path.is_file(), "task_exists": task_path.is_file()}
    train_cfg = parse_simple_yaml(train_path) if train_path.is_file() else {}
    task_cfg = parse_simple_yaml(task_path) if task_path.is_file() else {}

    keys = ["horizon", "n_obs_steps", "n_action_steps", "n_latency_steps", "dataset_obs_steps"]
    rep["resolved"] = {k: resolve(train_cfg.get(k), train_cfg) for k in keys}
    rep["resolved"]["crop_shape"] = resolve(dig(train_cfg, "policy.crop_shape"), train_cfg)
    rep["resolved"]["batch_size"] = resolve(dig(train_cfg, "dataloader.batch_size"), train_cfg)
    rep["resolved"]["num_epochs"] = resolve(dig(train_cfg, "training.num_epochs"), train_cfg)
    rep["resolved"]["max_train_steps"] = resolve(dig(train_cfg, "training.max_train_steps"), train_cfg)
    rep["resolved"]["obs_encoder_group_norm"] = resolve(dig(train_cfg, "policy.obs_encoder_group_norm"), train_cfg)
    for k in ("n_layer", "n_cond_layers", "n_head", "n_emb"):
        rep["resolved"][f"dit_{k}"] = resolve(dig(train_cfg, f"policy.{k}"), train_cfg)
    rep["resolved"]["noise_scheduler"] = dig(train_cfg, "policy.noise_scheduler._target_")
    rep["resolved"]["policy_target"] = dig(train_cfg, "policy._target_")

    action_cfg = dig(task_cfg, "shape_meta.action.shape")
    rep["task"] = {
        "task_name": task_cfg.get("name"),
        "action_shape": action_cfg,
        "action_lerobot_keys": [k.strip() for k in re.findall(r"'([^']+)'", dig(task_cfg, "shape_meta.action.lerobot_keys") or "")],
        "obs_keys": [], "env_runner": {},
    }
    obs_block = dig(task_cfg, "shape_meta.obs")
    if isinstance(obs_block, dict):
        for k, v in obs_block.items():
            if k == "__list__" or not isinstance(v, dict):
                continue
            rep["task"]["obs_keys"].append({
                "key": k, "shape": resolve(v.get("shape"), train_cfg),
                "type": v.get("type", "low_dim"),
                "lerobot_keys": re.findall(r"'([^']+)'", v.get("lerobot_keys") or ""),
            })
    for k in ("max_steps", "n_test", "fps", "n_envs"):
        rep["task"]["env_runner"][k] = dig(task_cfg, f"env_runner.{k}")
    rep["task"]["env_kwargs"] = {
        k: dig(task_cfg, f"env_runner.env_kwargs.{k}")
        for k in ("style_ids", "layout_ids", "layout_and_style_ids", "clutter_mode", "obj_instance_split")
    }
    # 视觉编码器来源（源码级证据）
    pol = args.dp_repo / "diffusion_policy" / "policy" / "diffusion_transformer_hybrid_image_policy.py"
    rep["policy_source"] = {"path": str(pol), "exists": pol.is_file(), "evidence": {}}
    if pol.is_file():
        ptext = pol.read_text(encoding="utf-8")
        for name, pat in [
            ("default_vision_backbone", r"vision_backbone=\"resnet18\""),
            ("lang_emb_branch", r"if \"lang_emb\" in obs_config\['low_dim'\]"),
            ("fiLM_backbone_class", r"core_kwargs\.backbone_class = \"ResNet18ConvFiLM\""),
            ("core_class", r"core_class = \"VisualCoreLanguageConditioned\""),
            ("robomimic_config_source", r"get_robomimic_config\(\s*\n?\s*algo_name='bc_rnn'"),
        ]:
            rep["policy_source"]["evidence"][name] = bool(re.search(pat, ptext))
    lds = args.dp_repo / "diffusion_policy" / "dataset" / "lerobot_dataset.py"
    rep["dataset_source"] = {"path": str(lds), "exists": lds.is_file()}
    if lds.is_file():
        ltext = lds.read_text(encoding="utf-8")
        rep["dataset_source"]["inline_assertions"] = re.findall(
            r"(abs_action is not supported[^\"]*)", ltext)
        rep["dataset_source"]["lang_encoder"] = bool(re.search(r"LangUtils\.LangEncoder", ltext))

    # 2) 编码器结构
    rep["encoder"] = (analyze_blocks_static(args.robomimic_repo) if args.no_torch
                      else analyze_blocks_native(args.robomimic_repo))

    # 3) horizon
    rep["horizons"] = parse_task_horizons(args.robocasa_repo)

    # 4) 显存
    bs = rep["resolved"].get("batch_size") or 192
    try:
        bs = int(bs)
    except Exception:
        bs = 192
    rep["memory"] = estimate_memory(args.dp_repo, bs, train_cfg)

    # ---- 输出 ----
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baseline_summary.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    L = ["# 阶段 1 官方 Diffusion Policy 基线审计\n"]
    r = rep["resolved"]
    L.append("## 1. 解析后的配置（源码级）\n")
    L.append(f"- 配置：`{args.config_name}` + `task/{args.task_config}`")
    L.append(f"- 输入 shape：obs 键 {len(rep['task']['obs_keys'])} 个，"
             f"action shape = {rep['task']['action_shape']}")
    L.append("")
    L.append("| 键 | 值 |")
    L.append("|---|---|")
    for k in ("horizon", "n_obs_steps", "n_action_steps", "crop_shape", "batch_size",
              "num_epochs", "max_train_steps", "obs_encoder_group_norm",
              "dit_n_layer", "dit_n_cond_layers", "dit_n_head", "dit_n_emb"):
        L.append(f"| {k} | {r.get(k)} |")
    L.append("")
    L.append("| obs 键 | shape | type | lerobot 键 |")
    L.append("|---|---|---|---|")
    for o in rep["task"]["obs_keys"]:
        L.append(f"| {o['key']} | {o['shape']} | {o['type']} | {', '.join(o['lerobot_keys']) or '-'} |")
    L.append("")
    L.append(f"动作向量 lerobot 键：`{rep['task']['action_lerobot_keys']}`\n")

    e = rep["encoder"]
    L.append("## 2. ResNet18ConvFiLM 的 block 结构\n")
    L.append(f"- 方法：**{e.get('method')}**")
    L.append(f"- conv block 数：**{e.get('n_conv_blocks')}**，通道：`{e.get('channels')}`")
    L.append(f"- 每个 block 后接 FiLM：**{e.get('film_layers')}** 个，语言维度 {e.get('film_lang_dim')}")
    for size, info in (e.get("sizes") or {}).items():
        L.append(f"- 输入 {size}×{size}：base_block 输出 {info.get('base_block_out')}，"
                 f"最终特征 {info.get('final_out')}")
    if e.get("evidence"):
        L.append("\n源码证据：")
        for ev in e["evidence"]:
            L.append(f"- `{ev.get('pattern')}` → {ev.get('found')}")
    L.append("")

    h = rep["horizons"]
    L.append("## 3. 每任务 horizon\n")
    if h.get("status") == "ok":
        L.append(f"- 任务数：**{h['n_tasks']}**，范围 **{h['min']}–{h['max']}** 步，"
                 f"全部为 150 的倍数：**{h['is_multiple_of_150']}**")
        L.append(f"- 分布（步数: 任务数）：`{h['distribution']}`")
        L.append("- 注意：配置文件里的 `max_steps` 会被 `eval_robocasa.py` 的 "
                 "`get_task_horizon(task)` 覆盖。")
    else:
        L.append(f"- 跳过：{h.get('reason')}")
    L.append("")

    m = rep["memory"]
    L.append("## 4. 显存与评测开销\n")
    L.append(f"- 官方建议：训练 ≥ **{m['official']['train_gpu_min_gb']} GB**"
             f"（推荐 {m['official']['train_gpu_recommended_gb']} GB+）；"
             f"推理 ≥ {m['official']['inference_gpu_min_gb']} GB")
    L.append(f"- 评测实例：{m['official']['eval_instance']}")
    L.append(f"- 本地实测：{m.get('measured') or m.get('measured_peak_gb_probe') or '未实测'}")
    L.append(f"- {m['analytic_note']}\n")

    ok = (rep["paths"]["train_exists"] and rep["paths"]["task_exists"])
    L.append("## 5. 判定\n")
    L.append(f"**{'基线配置已确认，可进入 dry run' if ok else '配置文件缺失，需先克隆官方 fork'}**")
    (args.out / "baseline_report.md").write_text("\n".join(L), encoding="utf-8")

    print(json.dumps({k: rep["resolved"].get(k) for k in
                      ("horizon", "n_obs_steps", "n_action_steps", "crop_shape", "batch_size")},
                     ensure_ascii=False))
    print("encoder:", rep["encoder"].get("method"), rep["encoder"].get("channels"))
    print("horizons:", h.get("status"), h.get("min"), h.get("max"))
    print("报告：", args.out / "baseline_report.md")


if __name__ == "__main__":
    main()
