#!/usr/bin/env python3
"""审计 RoboCasa365 的 style_id 在冻结 DINOv2 特征里是否可分。

不训练策略，只回答：
  1. 同 style / 异 style 特征距离是否有差；
  2. 1-NN 能否从单帧预测 style；
  3. style 原型在视觉特征空间里应聚成多少组（给粗域划分提供依据）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dacsc.dinov2_encoder import load_dinov2  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_frame(path: Path, frac: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        raise RuntimeError(f"无法读取视频帧数: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(n - 1, max(0, int(frac * (n - 1)))))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"无法读取帧: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    c = 224
    y, x = (h - c) // 2, (w - c) // 2
    if h < c or w < c:
        rgb = cv2.resize(rgb, (c, c), interpolation=cv2.INTER_LINEAR)
    else:
        rgb = rgb[y:y + c, x:x + c]
    x = rgb.astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def pair_stats(feat: np.ndarray, labels: np.ndarray):
    sim = feat @ feat.T
    same, diff = [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            (same if labels[i] == labels[j] else diff).append(float(sim[i, j]))
    return {
        "same_cos": float(np.mean(same)) if same else 0.0,
        "diff_cos": float(np.mean(diff)) if diff else 0.0,
        "same_minus_diff": (float(np.mean(same) - np.mean(diff))
                            if same and diff else 0.0),
        "n_same_pairs": len(same), "n_diff_pairs": len(diff),
    }


def knn_style(feat: np.ndarray, labels: np.ndarray):
    sim = feat @ feat.T
    np.fill_diagonal(sim, -1e9)
    pred = labels[np.argmax(sim, axis=1)]
    acc = float((pred == labels).mean())
    cnt = Counter(labels.tolist())
    chance = float(sum(v * v for v in cnt.values()) / (len(labels) ** 2))
    return {"knn1_acc": acc, "chance": chance, "lift": acc - chance}


def silhouette(feat: np.ndarray, labels: np.ndarray):
    dist = 1.0 - feat @ feat.T
    vals = []
    for i, yi in enumerate(labels):
        same = np.where(labels == yi)[0]
        same = same[same != i]
        if len(same) == 0:
            continue
        a = float(dist[i, same].mean())
        b = min(float(dist[i, np.where(labels == y)[0]].mean())
                for y in np.unique(labels) if y != yi and np.any(labels == y))
        vals.append((b - a) / max(a, b, 1e-8))
    return float(np.mean(vals)) if vals else 0.0


def kmeans_cosine(x: np.ndarray, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(30):
        centers = x[rng.choice(len(x), size=k, replace=False)].copy()
        for _ in range(200):
            centers = centers / np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-8)
            assign = np.argmax(x @ centers.T, axis=1)
            new = np.stack([x[assign == j].mean(0) if np.any(assign == j) else centers[j]
                            for j in range(k)])
            if np.allclose(new, centers, atol=1e-7):
                break
            centers = new
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-8)
        inertia = float((1.0 - np.max(x @ centers.T, axis=1)).sum())
        if best is None or inertia < best[0]:
            best = (inertia, assign.copy(), centers.copy())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True,
                    help="lerobot 数据集根目录，可重复传入")
    ap.add_argument("--view", default="robot0_agentview_left")
    ap.add_argument("--frame-frac", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/runs/style_separability.json"))
    args = ap.parse_args()

    meta, paths = [], []
    for root in map(Path, args.root):
        for p in sorted((root / "extras").glob("*/ep_meta.json")):
            d = json.loads(p.read_text())
            ep = int(p.parent.name.split("_")[-1])
            video = root / "videos" / "chunk-000" / f"observation.images.{args.view}" / f"episode_{ep:06d}.mp4"
            if not video.exists():
                continue
            meta.append({"style": int(d["style_id"]), "layout": int(d["layout_id"]),
                         "task": str(d.get("lang", "")), "video": str(video)})
            paths.append(video)
    if not meta:
        raise SystemExit("没有找到样本")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_dinov2("dinov2_vits14", pretrained=True).to(device).eval()
    feats = []
    bs = 32
    for i in range(0, len(paths), bs):
        batch = torch.stack([torch.from_numpy(read_frame(p, args.frame_frac))
                             for p in paths[i:i + bs]]).float().to(device)
        with torch.no_grad():
            out = net.forward_features(batch)
            f = out["x_norm_patchtokens"].mean(dim=1)
            f = torch.nn.functional.normalize(f.float(), dim=1)
        feats.append(f.cpu().numpy())
        print(f"encoded {min(i + bs, len(paths))}/{len(paths)}", flush=True)
    feat = np.concatenate(feats, 0)
    styles = np.asarray([m["style"] for m in meta])
    layouts = np.asarray([m["layout"] for m in meta])

    # layout 残差化：先减去每个 layout 的均值，再看 style 是否仍可分。
    resid = feat.copy()
    for lay in np.unique(layouts):
        m = layouts == lay
        if m.sum() >= 2:
            resid[m] = feat[m] - feat[m].mean(0, keepdims=True)
    resid /= np.linalg.norm(resid, axis=1, keepdims=True).clip(1e-8)

    # 同一 layout 内的 style 对照（避免 layout 信号冒充 style）。
    lay_same, lay_diff = [], []
    for lay in np.unique(layouts):
        idx = np.where(layouts == lay)[0]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                v = float(feat[idx[a]] @ feat[idx[b]])
                (lay_same if styles[idx[a]] == styles[idx[b]] else lay_diff).append(v)

    style_cent, style_ids = [], []
    for s in sorted(np.unique(styles)):
        style_cent.append(feat[styles == s].mean(0))
        style_ids.append(int(s))
    style_cent = np.stack(style_cent)
    style_cent /= np.linalg.norm(style_cent, axis=1, keepdims=True).clip(1e-8)
    inertia, assign, centers = kmeans_cosine(style_cent, min(args.k, len(style_cent)))
    clusters = defaultdict(list)
    for s, a in zip(style_ids, assign):
        clusters[int(a)].append(int(s))

    result = {
        "n_samples": len(meta), "n_styles": len(style_ids),
        "n_layouts": int(len(np.unique(layouts))),
        "style_pair_stats": pair_stats(feat, styles),
        "style_knn1": knn_style(feat, styles),
        "style_silhouette": silhouette(feat, styles),
        "layout_knn1": knn_style(feat, layouts),
        "within_layout_pair_stats": {
            "same_cos": float(np.mean(lay_same)) if lay_same else 0.0,
            "diff_cos": float(np.mean(lay_diff)) if lay_diff else 0.0,
            "same_minus_diff": (float(np.mean(lay_same) - np.mean(lay_diff))
                                if lay_same and lay_diff else 0.0),
            "n_same_pairs": len(lay_same), "n_diff_pairs": len(lay_diff),
        },
        "layout_residual_style_pair_stats": pair_stats(resid, styles),
        "layout_residual_style_knn1": knn_style(resid, styles),
        "layout_residual_style_silhouette": silhouette(resid, styles),
        "kmeans_k": int(args.k), "kmeans_inertia": inertia,
        "style_clusters": {str(k): v for k, v in sorted(clusters.items())},
        "style_counts": {str(int(k)): int(v) for k, v in sorted(Counter(styles).items())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
