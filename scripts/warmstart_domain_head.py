#!/usr/bin/env python3
"""Train frozen-DINOv2 right-agentview domain heads from the audit cache."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_masks(split):
    tr = split == "train"
    va = split == "val"
    return tr, va


def train_head(X, y, episode, split, num_classes, steps=500):
    tr, va = split_masks(split)
    classes = sorted(set(y[tr].tolist()) & set(y[va].tolist()))
    cmap = {c: i for i, c in enumerate(classes)}
    keep = np.array([int(v) in cmap for v in y])
    tr = tr & keep
    va = va & keep
    ytr = torch.tensor([cmap[int(v)] for v in y[tr]])
    yva = torch.tensor([cmap[int(v)] for v in y[va]])
    Xtr = torch.from_numpy(X[tr]).float()
    Xva = torch.from_numpy(X[va]).float()
    mean = Xtr.mean(0)
    std = Xtr.std(0).clamp_min(1e-6)
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std
    c = len(classes)
    model = nn.Linear(X.shape[1], c)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    counts = torch.bincount(ytr, minlength=c).float()
    weights = ytr.numel() / (c * counts.clamp_min(1.0))
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr), ytr, weight=weights)
        loss.backward()
        opt.step()
    with torch.no_grad():
        prob = F.softmax(model(Xva), dim=1).numpy()
    pred = prob.argmax(1)
    def metrics(true, pred):
        cm = np.zeros((c, c), dtype=np.int64)
        for t, p in zip(true, pred):
            cm[int(t), int(p)] += 1
        tp = np.diag(cm).astype(float)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        rec = tp / np.maximum(tp + fn, 1)
        pre = tp / np.maximum(tp + fp, 1)
        f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-9)
        support = cm.sum(1) > 0
        return {"balanced_acc": float(rec[support].mean()) if support.any() else 0.0,
                "macro_f1": float(f1[support].mean()) if support.any() else 0.0,
                "support": cm.sum(1).tolist(), "cm": cm.tolist()}
    frame = metrics(yva.tolist(), pred.tolist())
    ep_pred = {}
    for e, p in zip(episode[va].tolist(), prob):
        ep_pred.setdefault(int(e), []).append(p)
    ep_true, ep_prob = [], []
    for e, ps in sorted(ep_pred.items()):
        ep_true.append(cmap[int(y[episode == e][0])])
        ep_prob.append(np.mean(ps, axis=0))
    ep_pred_lbl = np.argmax(np.stack(ep_prob), axis=1)
    ep = metrics(ep_true, ep_pred_lbl.tolist())
    return model, mean, std, {"classes": classes, "frame": frame, "episode": ep,
                              "n_train_frames": int(tr.sum()),
                              "n_val_frames": int(va.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    default=Path("/hy-tmp/runs/pure_dino_style_features_full.pt"))
    ap.add_argument("--out", type=Path,
                    default=Path("/hy-tmp/runs/domain_head_warmstart.pt"))
    ap.add_argument("--view", default="robot0_agentview_right")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--layout-cache", type=Path, default=None)
    ap.add_argument("--base-checkpoint", type=Path, default=None)
    args = ap.parse_args()
    obj = torch.load(args.cache, map_location="cpu", weights_only=False)
    def extract(cache_obj):
        idx = [i for i, m in enumerate(cache_obj["meta"]) if m["view"] == args.view]
        meta = [cache_obj["meta"][i] for i in idx]
        X = cache_obj["patch"][idx].numpy()
        style = np.array([m["style"] for m in meta])
        layout = np.array([m["layout"] for m in meta])
        split = np.array([m["split"] for m in meta])
        episode = np.array([f"{m['task']}:{m['episode']}" for m in meta])
        ep_map = {k: i for i, k in enumerate(sorted(set(episode.tolist())))}
        episode = np.array([ep_map[k] for k in episode], dtype=np.int64)
        return X, style, layout, split, episode
    X, style, layout, split, episode = extract(obj)
    labels = {
        "layout10": np.clip((layout - 11) // 5, 0, 9),
        "coarse3": np.clip((style - 11) // 20, 0, 2),
    }
    out = {"feature": args.view, "feature_type": "patch_mean"}
    summary = {}
    if args.layout_cache is not None:
        base = torch.load(args.base_checkpoint or args.out, map_location="cpu",
                          weights_only=False)
        out["coarse3"] = base["coarse3"]
        out["mean"] = base.get("mean", base["coarse3"].get("mean"))
        out["std"] = base.get("std", base["coarse3"].get("std"))
        layout_obj = torch.load(args.layout_cache, map_location="cpu", weights_only=False)
        Xl, style_l, layout_l, split_l, ep_l = extract(layout_obj)
        y = np.clip((layout_l - 11) // 5, 0, 9)
        model, mean, std, metrics = train_head(Xl, y, ep_l, split_l, 10, steps=args.steps)
        out["layout10"] = {"weight": model.weight.detach().cpu(),
                           "bias": model.bias.detach().cpu(),
                           "mean": mean.cpu(), "std": std.cpu()}
        summary["layout10"] = metrics
        print("layout10", "frame_bal", round(metrics["frame"]["balanced_acc"], 4),
              "episode_bal", round(metrics["episode"]["balanced_acc"], 4))
    else:
        for name, y in labels.items():
            C = int(max(y.max() + 1, {"layout10": 10, "coarse3": 3}[name]))
            model, mean, std, metrics = train_head(
                X, y, episode, split, C, steps=args.steps)
            out[name] = {"weight": model.weight.detach().cpu(),
                         "bias": model.bias.detach().cpu(),
                         "mean": mean.cpu(), "std": std.cpu()}
            summary[name] = metrics
            print(name, "frame_bal", round(metrics["frame"]["balanced_acc"], 4),
                  "episode_bal", round(metrics["episode"]["balanced_acc"], 4),
                  "episode_f1", round(metrics["episode"]["macro_f1"], 4))
        out["mean"] = mean.cpu()
        out["std"] = std.cpu()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    (args.out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
