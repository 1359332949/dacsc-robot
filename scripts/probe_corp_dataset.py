"""验证 CorpRoboCasaDataset：严格 split、batch 形状、normalizer。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, "/root/dacsc-robot")
from dacsc.corp_data import (  # noqa: E402
    CorpRoboCasaDataset, LinearNormalizer, collate_corp,
)

ROOT = Path("/hy-tmp/datasets/robocasa365/v1.0/pretrain/atomic")
TASKS = ["CloseFridge", "OpenFridge", "CloseDrawer"]
PATHS = []
for t in TASKS:
    cands = sorted((ROOT / t).glob("*/lerobot"))
    PATHS.extend(str(c) for c in cands[:1])
print("paths:", PATHS)


def check(style_min, style_max, n=3):
    ds = CorpRoboCasaDataset(PATHS, style_min=style_min, style_max=style_max)
    styles = sorted({ds.flat[i][2] for i in range(min(len(ds), 5000))})
    assert all(style_min <= s <= style_max for s in styles), styles
    print(f"split [{style_min},{style_max}]: len={len(ds)} styles_seen={styles[:10]}...")
    batch = collate_corp([ds[i] for i in range(n)])
    print("  images:", {k: tuple(v.shape) for k, v in batch["images"].items()})
    print("  state:", tuple(batch["state"].shape),
          "action:", tuple(batch["action"].shape),
          "instr:", batch["instruction"][:2])
    return ds


train = check(11, 30)
val = check(31, 40)
test = check(41, 50)

norm = LinearNormalizer.fit(train, max_steps=128)
print("normalizer state_mean:", [round(float(x), 3) for x in norm.state_mean])

b = collate_corp([train[i] for i in range(2)])
ns = norm.normalize_state(b["state"])
na = norm.normalize_action(b["action"])
print("normalized state mean~0:", [round(float(x), 3) for x in ns.mean(0)])
print("normalized action range (identity):", round(float(na.min()), 3), round(float(na.max()), 3))
print("SPLIT + BATCH OK")
