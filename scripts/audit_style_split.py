#!/usr/bin/env python3
"""Audit leakage barriers for train 11-30 / val 31-40 / test 41-50.

This script verifies both sides of the split:
  * the policy's train/val/test index sets are disjoint and style-correct;
  * the domain-balanced training sampler cannot emit val/test indices;
  * every split's episode/task/layout support is reported explicitly.

It does not train a policy or evaluate an action metric.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "repos/diffusion_policy/diffusion_policy/config"


def main():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="train_dinov2_flow",
            overrides=["task=robocasa/pretrain_atomic61",
                       "dataloader.num_workers=0",
                       "val_dataloader.num_workers=0",
                       "logging.mode=disabled"])
    OmegaConf.resolve(cfg)
    ds = hydra.utils.instantiate(cfg.task.dataset)

    split_indices = {
        "train": list(ds._train_indices),
        "val": list(ds._val_indices),
        "test": list(ds._test_indices),
    }
    split_ranges = {
        "train": set(range(ds._train_style_min, ds._train_style_max + 1)),
        "val": set(range(ds._val_style_min, ds._val_style_max + 1)),
        "test": set(range(ds._test_style_min, ds._test_style_max + 1)),
    }
    reserved_styles = set(range(ds._reserved_style_min,
                                ds._reserved_style_max + 1))

    # Episode-level metadata support, independent of frame sampling.
    style_episodes = defaultdict(list)
    for sub_idx, sub in enumerate(ds.datasets):
        task = sub.dataset_path.parents[1].name
        for episode, (style, layout) in getattr(sub, "domain_map", {}).items():
            if style is None:
                continue
            style_episodes[int(style)].append({
                "key": f"{sub_idx}:{int(episode)}",
                "task": task,
                "layout": None if layout is None else int(layout),
            })

    split_style_indices = {}
    for name, idx in split_indices.items():
        index_set = set(idx)
        styles = {
            style for style, values in ds._style_to_indices.items()
            if not index_set.isdisjoint(values)
        }
        assert styles <= split_ranges[name], (
            f"{name} indices contain out-of-range styles: "
            f"{sorted(styles - split_ranges[name])}")
        assert styles == split_ranges[name], (
            f"{name} is missing styles: {sorted(split_ranges[name] - styles)}")
        split_style_indices[name] = sorted(styles)

    disjoint = {}
    names = list(split_indices)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            disjoint[f"{left}_{right}"] = set(split_indices[left]).isdisjoint(
                split_indices[right])
            assert disjoint[f"{left}_{right}"], f"{left}/{right} indices overlap"

    sampler = ds.get_dataset_sampler()
    sampler._build()
    sampler_all = set(int(x) for x in sampler._all)
    train_set = set(split_indices["train"])
    sampler_style_sets = {}
    for group, values in sampler._groups.items():
        group_set = set(int(x) for x in values)
        styles = sorted({
            style for style, style_idx in ds._style_to_indices.items()
            if not group_set.isdisjoint(style_idx)
        })
        sampler_style_sets[str(group)] = styles
    sampler_ok = sampler_all <= train_set
    assert sampler_ok, "training sampler contains indices outside train split"
    assert all(set(v) <= set(range(ds._train_style_min,
                                    ds._train_style_max + 1))
               for v in sampler_style_sets.values())

    out = {
        "domain_id_min": ds._domain_id_min,
        "normalizer": {
            "train_only": bool(ds._train_only_normalizer),
            "max_steps": int(ds._normalizer_max_steps),
        },
        "disjoint_indices": disjoint,
        "reserved_styles": sorted(reserved_styles),
        "training_sampler": {
            "allowed_styles": sorted(sampler.allowed_styles),
            "groups_to_styles": sampler_style_sets,
            "n_candidate_steps": len(sampler_all),
            "all_candidates_in_train_split": sampler_ok,
        },
        "splits": {},
    }
    for name, idx in split_indices.items():
        styles = split_style_indices[name]
        episodes = [
            episode
            for style in styles
            for episode in style_episodes.get(style, [])
        ]
        episode_keys = [e["key"] for e in episodes]
        assert len(episode_keys) == len(set(episode_keys)), (
            f"{name} episode support contains duplicates")
        tasks = {e["task"] for e in episodes}
        layouts = {e["layout"] for e in episodes if e["layout"] is not None}
        per_style = {}
        for style in styles:
            support = style_episodes.get(style, [])
            per_style[str(style)] = {
                "episodes": len(support),
                "tasks": len({e["task"] for e in support}),
                "layouts": len({e["layout"] for e in support
                                if e["layout"] is not None}),
                "index_steps": len(ds._style_to_indices.get(style, [])),
                "included_index_steps": len(
                    set(ds._style_to_indices.get(style, [])) & set(idx)),
            }
        out["splits"][name] = {
            "style_min": int(min(styles)),
            "style_max": int(max(styles)),
            "styles": styles,
            "episodes": len(episodes),
            "tasks": len(tasks),
            "layouts": len(layouts),
            "index_steps": len(idx),
            "per_style": per_style,
        }

    out["acceptance"] = {
        "train_sampler_only_train_styles": sampler_ok,
        "val_only_styles_31_40": split_style_indices["val"] == list(range(31, 41)),
        "test_only_styles_41_50": split_style_indices["test"] == list(range(41, 51)),
        "train_val_test_indices_disjoint": all(disjoint.values()),
        "normalizer_train_only": bool(ds._train_only_normalizer),
        "reserved_styles_excluded": all(
            set(split_indices[name]).isdisjoint(
                set().union(*[ds._style_to_indices.get(s, [])
                              for s in reserved_styles]))
            for name in split_indices),
    }
    out["acceptance"]["passed"] = all(out["acceptance"].values())

    target = ROOT / "audit/style_split_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {target}")
    if not out["acceptance"]["passed"]:
        raise SystemExit("style split audit failed")


if __name__ == "__main__":
    main()
