#!/usr/bin/env python3
"""Pure-vision, episode-level style audit with frozen DINOv2.

No language, FiLM, policy, gate, dropout, or task-conditioned encoder is used.
The script reads RoboCasa365 LeRobot episodes directly:
  extras/episode_XXXXXX/ep_meta.json -> style_id / layout_id
  videos/.../observation.images.*/episode_XXXXXX.mp4
and extracts frozen DINOv2 ViT-S/14 CLS + patch-mean features.

The main probe is trained/evaluated with an episode-level split, so adjacent
frames from the same episode never cross train/val.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
VIEWS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
VIDEO_PREFIX = "observation.images."
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def scan(root: Path):
    rows = []
    for lerobot in sorted(root.glob("v1.0/pretrain/atomic/*/*/lerobot")):
        task = lerobot.parents[1].name
        ep_len = {}
        ep_file = lerobot / "meta" / "episodes.jsonl"
        if ep_file.is_file():
            for line in ep_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                ep_len[int(d["episode_index"])] = int(d.get("length", 0))
        for ep_dir in sorted((lerobot / "extras").glob("episode_*")):
            m = re.fullmatch(r"episode_(\d+)", ep_dir.name)
            if not m:
                continue
            idx = int(m.group(1))
            meta_f = ep_dir / "ep_meta.json"
            if not meta_f.is_file():
                continue
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            style, layout = meta.get("style_id"), meta.get("layout_id")
            if style is None or layout is None:
                continue
            videos = {}
            ok = True
            for view in VIEWS:
                hits = list((lerobot / "videos").glob(
                    f"chunk-*/{VIDEO_PREFIX}{view}/episode_{idx:06d}.mp4"))
                if not hits:
                    ok = False
                    break
                videos[view] = str(hits[0])
            if not ok:
                continue
            rows.append({"task": task, "episode": idx, "style": int(style),
                         "layout": int(layout), "length": ep_len.get(idx, 0),
                         "lerobot": str(lerobot), "videos": videos})
    return rows


def select_episodes(rows, per_group: int, seed: int, balance_by: str = "style"):
    buckets = defaultdict(list)
    for r in rows:
        if balance_by == "layout":
            key = r["layout"]
        else:
            key = r["style"]
        buckets[key].append(r)
    rng = random.Random(seed)
    selected = []
    for group, eps in sorted(buckets.items()):
        sub = defaultdict(list)
        for e in eps:
            # For layout balance, diversify style/task; for style balance,
            # diversify layout/task (the original behavior).
            if balance_by == "layout":
                sub[(e["style"], e["task"])].append(e)
            else:
                sub[(e["layout"], e["task"])].append(e)
        for g in sub.values():
            rng.shuffle(g)
        keys = sorted(sub)
        picked = []
        while len(picked) < min(per_group, len(eps)):
            for k in keys:
                if sub[k] and len(picked) < per_group:
                    picked.append(sub[k].pop())
            if all(not sub[k] for k in keys):
                break
        rng.shuffle(picked)
        n_val = 1 if len(picked) >= 2 else 0
        if len(picked) >= 3:
            n_val = max(1, int(round(0.3 * len(picked))))
        for i, e in enumerate(picked):
            e = dict(e)
            e["split"] = "val" if i < n_val else "train"
            selected.append(e)
    return selected


def save_samples(selected, out_dir: Path, n_styles: int = 3, n_frames: int = 3):
    groups = defaultdict(list)
    for e in selected:
        groups[(e["task"], e["layout"])].append(e)
    best = None
    for key, eps in groups.items():
        if len({e["style"] for e in eps}) >= n_styles:
            best = (key, eps)
            break
    if best is None:
        return None
    (task, layout), eps = best
    chosen = []
    for style in sorted({e["style"] for e in eps})[:n_styles]:
        chosen.append(next(e for e in eps if e["style"] == style))
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for e in chosen:
        for idx, views in sample_video_frames(e, n_frames):
            frame = views["robot0_agentview_left"]
            p = out_dir / f"{task}_layout{layout}_style{e['style']}_frame{idx}.png"
            cv2.imwrite(str(p), frame)
            cells.append((p, f"style {e['style']} | layout {layout} | {task} | f{idx}"))
    if cells:
        thumb = 256
        cols = n_styles
        rows = math.ceil(len(cells) / cols)
        sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 28)), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        for i, (p, label) in enumerate(cells):
            img = Image.open(p).convert("RGB").resize((thumb, thumb))
            r, c = divmod(i, cols)
            x, y = c * thumb, r * (thumb + 28)
            sheet.paste(img, (x, y))
            draw.text((x + 4, y + thumb + 4), label, fill=(0, 0, 0))
        sheet.save(out_dir / "contact_sheet.png")
    return {"task": task, "layout": layout,
            "styles": [e["style"] for e in chosen],
            "n_images": len(cells), "dir": str(out_dir)}


def sample_video_frames(ep, n_frames: int):
    caps = {v: cv2.VideoCapture(ep["videos"][v]) for v in VIEWS}
    totals = []
    for cap in caps.values():
        totals.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    total = min(t for t in totals if t > 0) if any(t > 0 for t in totals) else 0
    if total <= 0:
        for cap in caps.values():
            cap.release()
        return []
    n = min(n_frames, total)
    idxs = np.linspace(0.2 * (total - 1), 0.8 * (total - 1), n).round().astype(int)
    out = []
    for idx in idxs:
        frames = {}
        for v, cap in caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                break
            frames[v] = frame
        if len(frames) == len(VIEWS):
            out.append((int(idx), frames))
    for cap in caps.values():
        cap.release()
    return out


def preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if min(h, w) < 224:
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    else:
        y, x = (h - 224) // 2, (w - 224) // 2
        rgb = rgb[y:y + 224, x:x + 224]
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def build_dino(device):
    from dacsc.dinov2_encoder import load_dinov2
    model = load_dinov2("dinov2_vits14", pretrained=True).to(device).eval()
    return model


def extract(selected, cache: Path, n_frames: int, batch_size: int, device, reuse: bool):
    if reuse and cache.exists():
        obj = torch.load(cache, map_location="cpu", weights_only=False)
        print("loaded feature cache", cache)
        return obj
    model = build_dino(device)
    meta, patch, cls = [], [], []
    for i, ep in enumerate(selected, 1):
        frames = sample_video_frames(ep, n_frames)
        for frame_idx, views in frames:
            for view in VIEWS:
                meta.append({"task": ep["task"], "episode": ep["episode"],
                             "style": ep["style"], "layout": ep["layout"],
                             "split": ep["split"], "frame": frame_idx,
                             "view": view})
        # batch all views/frames of this episode
        imgs = [preprocess(views[v]) for _, views in frames for v in VIEWS]
        for start in range(0, len(imgs), batch_size):
            x = torch.stack(imgs[start:start + batch_size]).to(device)
            with torch.inference_mode():
                out = model.forward_features(x)
                patch.append(out["x_norm_patchtokens"].mean(dim=1).float().cpu())
                cls.append(out["x_norm_clstoken"].float().cpu())
        if i % 25 == 0:
            print(f"extracted {i}/{len(selected)} episodes", flush=True)
    obj = {"meta": meta, "patch": torch.cat(patch),
           "cls": torch.cat(cls)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, cache)
    print("wrote feature cache", cache)
    return obj


def stratified_split(episodes, test_frac=0.3, seed=0):
    """Already assigned by select_episodes; kept for clarity."""
    tr = [e for e in episodes if e["split"] == "train"]
    va = [e for e in episodes if e["split"] == "val"]
    return tr, va


def probe(X, y, episode, split, kind: str, seed: int):
    split = torch.as_tensor(split)
    y = torch.as_tensor(y)
    episode = torch.as_tensor(episode)
    if split.dtype == torch.bool:
        tr, va = split, ~split
    else:
        tr = split == "train"
        va = split == "val"
    classes = sorted(set(y[tr].tolist()) & set(y[va].tolist()))
    if len(classes) < 2:
        return None
    cmap = {c: i for i, c in enumerate(classes)}
    keep = torch.tensor([int(v) in cmap for v in y])
    tr = tr & keep
    va = va & keep
    ytr = torch.tensor([cmap[int(v)] for v in y[tr]])
    yva = torch.tensor([cmap[int(v)] for v in y[va]])
    Xtr, Xva = X[tr].float(), X[va].float()
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    c = len(classes)
    if kind == "linear":
        model = nn.Linear(X.shape[1], c)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
        steps = 300
    else:
        model = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(),
                              nn.Dropout(0.2), nn.Linear(256, c))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        steps = 500
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
    # frame-level
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
                "cm": cm.tolist(),
                "support": cm.sum(1).tolist()}
    frame_m = metrics(yva.tolist(), pred.tolist())
    # episode-level: average probabilities over val frames
    ep_pred = {}
    for e, p in zip(episode[va].tolist(), prob):
        ep_pred.setdefault(int(e), []).append(p)
    ep_true, ep_prob = [], []
    for e, ps in sorted(ep_pred.items()):
        ep_true.append(cmap[int(y[episode == e][0])])
        ep_prob.append(np.mean(ps, axis=0))
    ep_pred_lbl = np.argmax(np.stack(ep_prob), axis=1)
    ep_m = metrics(ep_true, ep_pred_lbl.tolist())
    return {"classes": classes, "frame": frame_m, "episode": ep_m,
            "n_train_frames": int(tr.sum()), "n_val_frames": int(va.sum()),
            "n_train_episodes": len(set(episode[tr].tolist())),
            "n_val_episodes": len(set(episode[va].tolist()))}


def feature_matrix(obj, mode: str):
    meta = obj["meta"]
    patch = obj["patch"]
    cls = obj["cls"]
    # rows are episode-major, view order fixed
    n = len(meta)
    for i, m in enumerate(meta):
        assert m["view"] == VIEWS[i % len(VIEWS)]
    n_frames = n // len(VIEWS)
    patch = patch.reshape(n_frames, len(VIEWS), -1)
    cls = cls.reshape(n_frames, len(VIEWS), -1)
    frame_meta = meta[::len(VIEWS)]
    if mode == "concat":
        X = patch.reshape(n_frames, -1)
    else:
        view_idx = VIEWS.index(mode.split(":")[0])
        X = patch[:, view_idx] if ":patch" in mode else cls[:, view_idx]
    return X, frame_meta




def write_summary_csv(results, path: Path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        path.unlink()
    fields = [
        "feature", "label", "classifier", "control_value",
        "episode_balanced_acc", "episode_macro_f1",
        "frame_balanced_acc", "frame_macro_f1",
        "n_train_episodes", "n_val_episodes",
        "n_train_frames", "n_val_frames",
        "episode_support", "frame_support",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            for key in ("episode_support", "frame_support"):
                row[key] = json.dumps(r.get(key), separators=(",", ":"))
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/hy-tmp/datasets/robocasa365"))
    ap.add_argument("--per-style", "--per-group", dest="per_style", type=int, default=10)
    ap.add_argument("--balance-by", choices=["style", "layout"], default="style",
                    help="balance sampling by the native style_id or layout_id")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache", type=Path, default=Path("/hy-tmp/runs/pure_dino_style_features.pt"))
    ap.add_argument("--out", type=Path, default=ROOT / "audit/pure_dino_style")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = scan(args.root)
    print(f"scanned {len(rows)} episodes with labels/videos")
    style_stats = {}
    for s in sorted({r["style"] for r in rows}):
        eps = [r for r in rows if r["style"] == s]
        style_stats[s] = {"episodes": len(eps),
                          "tasks": len({r["task"] for r in eps}),
                          "layouts": len({r["layout"] for r in eps})}
    selected = select_episodes(rows, args.per_style, seed=42, balance_by=args.balance_by)
    print(f"selected {len(selected)} episodes")
    sample_info = save_samples(selected, args.out / "samples")
    print("sample images", sample_info)
    split_stats = defaultdict(lambda: {"train": 0, "val": 0})
    for e in selected:
        split_stats[e["style"]][e["split"]] += 1
    with (args.out / "label_audit.json").open("w") as f:
        json.dump({"label_source": "extras/episode_*/ep_meta.json",
                   "n_episodes": len(rows), "styles": style_stats,
                   "selected_style_split": dict(split_stats)}, f, indent=2)
    with (args.out / "selected_episodes.jsonl").open("w") as f:
        for e in selected:
            f.write(json.dumps({k: e[k] for k in
                                ("task", "episode", "style", "layout", "split")}) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = extract(selected, args.cache, args.frames, args.batch_size, device, args.reuse)
    meta = obj["meta"]
    tr = np.array([m["split"] == "train" for m in meta[::len(VIEWS)]])
    style = np.array([m["style"] for m in meta[::len(VIEWS)]])
    layout = np.array([m["layout"] for m in meta[::len(VIEWS)]])
    task = np.array([m["task"] for m in meta[::len(VIEWS)]])
    ep_keys = [f"{m['task']}:{m['episode']}" for m in meta[::len(VIEWS)]]
    ep_map = {k: i for i, k in enumerate(sorted(set(ep_keys)))}
    episode = np.array([ep_map[k] for k in ep_keys], dtype=np.int64)
    task_names = sorted(set(task.tolist()))
    task_rank = {t: i for i, t in enumerate(task_names)}
    # Only native labels from ep_meta.json / episode metadata are audited.
    labels = {
        "style": style,
        "layout": layout,
        "task": np.array([task_rank[t] for t in task], dtype=np.int64),
    }
    split = np.array([m["split"] == "train" for m in meta[::len(VIEWS)]])
    support = {}
    for label_name, y in labels.items():
        support[label_name] = {}
        for cls in sorted(set(y.tolist())):
            msk = y == cls
            support[label_name][str(int(cls))] = {
                "train_episodes": len(set(episode[msk & split].tolist())),
                "val_episodes": len(set(episode[msk & ~split].tolist())),
                "train_frames": int((msk & split).sum()),
                "val_frames": int((msk & ~split).sum()),
            }
    # use the same row labels/split for each view; extract() rows are view-interleaved
    n_frames = len(meta) // len(VIEWS)
    assert len(style) == n_frames

    results = []
    features = {}
    for mode in ["robot0_agentview_left:patch", "robot0_agentview_right:patch",
                 "robot0_eye_in_hand:patch", "robot0_agentview_left:cls",
                 "robot0_agentview_right:cls", "robot0_eye_in_hand:cls",
                 "concat"]:
        X, frame_meta = feature_matrix(obj, mode)
        features[mode] = X
        for label_name, y in labels.items():
            for kind in ("linear", "mlp"):
                m = probe(X, torch.from_numpy(y), torch.from_numpy(episode),
                          torch.from_numpy(split), kind, seed=0)
                if m is None:
                    continue
                row = {"feature": mode, "label": label_name, "classifier": kind,
                       **{f"frame_{k}": v for k, v in m["frame"].items()},
                       **{f"episode_{k}": v for k, v in m["episode"].items()},
                       "n_train_frames": m["n_train_frames"],
                       "n_val_frames": m["n_val_frames"],
                       "n_train_episodes": m["n_train_episodes"],
                       "n_val_episodes": m["n_val_episodes"]}
                results.append(row)
                print(f"{mode:34s} {label_name:10s} {kind:6s} "
                      f"frame_bal={m['frame']['balanced_acc']:.3f} "
                      f"ep_bal={m['episode']['balanced_acc']:.3f}")

    # Controlled subsets keep the native style label; no synthetic regrouping.
    for control in ("same_task", "same_layout"):
        if control == "same_task":
            cand = Counter(task.tolist())
            best = max(task_names, key=lambda t: (len(set(style[task == t])), cand[t]))
            mask = task == best
            chosen = best
        else:
            cand = Counter(layout.tolist())
            best = max(sorted(set(layout.tolist())),
                       key=lambda l: (len(set(style[layout == l])), cand[l]))
            mask = layout == best
            chosen = int(best)
        X, _ = feature_matrix(obj, "robot0_agentview_left:patch")
        y = style[mask]
        e = episode[mask]
        s = split[mask]
        m = probe(X[mask], torch.from_numpy(y), torch.from_numpy(e),
                  torch.from_numpy(s), "linear", seed=0)
        if m is not None:
            row = {"feature": "robot0_agentview_left:patch",
                   "label": f"{control}_style", "classifier": "linear",
                   "control_value": chosen,
                   "classes": m["classes"],
                   **{f"frame_{k}": v for k, v in m["frame"].items()},
                   **{f"episode_{k}": v for k, v in m["episode"].items()},
                   "n_train_frames": m["n_train_frames"],
                   "n_val_frames": m["n_val_frames"],
                   "n_train_episodes": m["n_train_episodes"],
                   "n_val_episodes": m["n_val_episodes"]}
            results.append(row)
            print(f"{control:12s} {chosen} ep_bal={m['episode']['balanced_acc']:.3f}")

    with (args.out / "results.json").open("w") as f:
        json.dump({"selected_episodes": len(selected), "n_frames": n_frames,
                   "styles": style_stats, "split": dict(split_stats),
                   "label_support": support, "sample_images": sample_info,
                   "task_names": task_names, "results": results}, f, indent=2)
    write_summary_csv(results, args.out / "summary.csv")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
