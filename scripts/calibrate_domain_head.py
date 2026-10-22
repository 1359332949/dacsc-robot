#!/usr/bin/env python3
"""Fit per-head temperatures on the held-out audit episodes."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def extract(cache, view):
    obj = torch.load(cache, map_location="cpu", weights_only=False)
    idx = [i for i, m in enumerate(obj["meta"]) if m["view"] == view]
    meta = [obj["meta"][i] for i in idx]
    X = obj["patch"][idx].float()
    return X, meta


def fit_temp(logits, y):
    best = (1.0, float(F.cross_entropy(logits, y)))
    for T in np.geomspace(0.1, 20.0, 80):
        ce = float(F.cross_entropy(logits / float(T), y))
        if ce < best[1]:
            best = (float(T), ce)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("/hy-tmp/runs/domain_head_warmstart.pt"))
    ap.add_argument("--style-cache", type=Path,
                    default=Path("/hy-tmp/runs/pure_dino_style_features_full.pt"))
    ap.add_argument("--layout-cache", type=Path,
                    default=Path("/hy-tmp/runs/pure_dino_layout_balanced.pt"))
    ap.add_argument("--out", type=Path,
                    default=Path("/hy-tmp/runs/domain_head_warmstart_calibrated.pt"))
    ap.add_argument("--view", default="robot0_agentview_right")
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # coarse3: style-balanced cache
    X, meta = extract(args.style_cache, args.view)
    style = np.array([m["style"] for m in meta])
    split = np.array([m["split"] == "val" for m in meta])
    y = torch.tensor(np.clip((style - 11) // 20, 0, 2)[split])
    head = ckpt["coarse3"]
    mean = head.get("mean", ckpt["mean"]).float()
    std = head.get("std", ckpt["std"]).float()
    logits = ((X[split] - mean) / std) @ head["weight"].float().t() + head["bias"].float()
    T, ce = fit_temp(logits, y)
    head["temperature"] = T
    print("coarse3", "T", round(T, 3), "val_ce", round(ce, 4),
          "acc", round(float((logits.argmax(1) == y).float().mean()), 4))

    # layout10: layout-balanced cache
    X, meta = extract(args.layout_cache, args.view)
    layout = np.array([m["layout"] for m in meta])
    split = np.array([m["split"] == "val" for m in meta])
    y = torch.tensor(np.clip((layout - 11) // 5, 0, 9)[split])
    head = ckpt["layout10"]
    mean = head.get("mean", ckpt["mean"]).float()
    std = head.get("std", ckpt["std"]).float()
    logits = ((X[split] - mean) / std) @ head["weight"].float().t() + head["bias"].float()
    T, ce = fit_temp(logits, y)
    head["temperature"] = T
    print("layout10", "T", round(T, 3), "val_ce", round(ce, 4),
          "acc", round(float((logits.argmax(1) == y).float().mean()), 4))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
