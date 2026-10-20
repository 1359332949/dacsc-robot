#!/usr/bin/env python3
"""Linear-probe audit of ResNet18 stage-end features for style/layout/task.

This is the same feature space used by the faithful CNN DomainDrop:
ResNet18ConvFiLM stage-end GAP features at FiLM layers 1/3/5/7.

It compares:
  * current 3 contiguous style groups;
  * KMeans style clusters;
  * layout domains;
  * task domains;
  * style after layout residualization.

Only linear probes are used, so a low score means the label is not linearly
separable in this representation (not that a nonlinear classifier cannot work).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / "repos/diffusion_policy/diffusion_policy/config"


class TaskLabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.task_map = {id(ds): i for i, ds in enumerate(dataset.datasets)}
        self.task_names = [getattr(ds, "dataset_path", str(i))
                           for i, ds in enumerate(dataset.datasets)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        ds, _, _ = self.dataset.sample_step(idx)
        sample["task_id"] = torch.tensor(self.task_map[id(ds)], dtype=torch.long)
        return sample


def compose_cfg(batch: int):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = hydra.compose(
            config_name="train_official_cnn_dropout",
            overrides=["task=robocasa/pretrain_atomic61",
                       f"dataloader.batch_size={batch}",
                       f"task.dataset.domain_batch_size={batch}",
                       "task.dataset.domain_group_size=20",
                       "task.dataset.domain_samples_per_group=32",
                       "task.dataset.domain_max_groups_per_batch=3",
                       "dataloader.num_workers=4",
                       "val_dataloader.num_workers=1",
                       "logging.mode=disabled",
                       "policy.cnn_dropout_mode=none"])
    OmegaConf.resolve(cfg)
    return cfg


def collect(cfg, batches: int, device: torch.device):
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    wrapped = TaskLabeledDataset(dataset)
    sampler = dataset.get_dataset_sampler()
    loader = DataLoader(wrapped, batch_size=cfg.dataloader.batch_size,
                        sampler=sampler, num_workers=cfg.dataloader.num_workers,
                        pin_memory=True)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    policy.eval()
    policy.obs_encoder.to(device)

    try:
        from robomimic.models.base_nets import ResNet18ConvFiLM
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("robomimic ResNet18ConvFiLM unavailable") from exc

    backbones = []
    for name, module in policy.obs_encoder.named_modules():
        if isinstance(module, ResNet18ConvFiLM):
            view = name.split(".")[1] if "." in name else name
            backbones.append((view, module))
    stage_ends = [1, 3, 5, 7]
    feats = defaultdict(list)
    handles = []
    for view, backbone in backbones:
        for idx in stage_ends:
            def hook(module, inputs, output, view=view, idx=idx):
                feats[(view, idx)].append(
                    output.detach().mean(dim=(2, 3)).float().cpu())
            handles.append(backbone._film_layers[idx].register_forward_hook(hook))

    style, layout, task = [], [], []
    To = int(policy.n_obs_steps)
    it = iter(loader)
    for _ in range(batches):
        batch = next(it)
        nobs = policy.normalizer.normalize(batch["obs"])
        this_nobs = {k: v[:, :To].reshape(-1, *v.shape[2:]).to(device)
                     for k, v in nobs.items()}
        with torch.no_grad():
            policy.obs_encoder(this_nobs)
        style.append(batch["domain_id"].repeat_interleave(To).cpu())
        layout.append(batch["layout_id"].repeat_interleave(To).cpu())
        task.append(batch["task_id"].repeat_interleave(To).cpu())
    for h in handles:
        h.remove()

    out = {k: torch.cat(v, dim=0) for k, v in feats.items()}
    labels = {"style": torch.cat(style), "layout": torch.cat(layout),
              "task": torch.cat(task)}
    return out, labels, wrapped.task_names, policy


def stratified_split(y: torch.Tensor, test_frac: float = 0.3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    train, test = [], []
    for c in y.unique():
        idx = torch.nonzero(y == c, as_tuple=False).view(-1)
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        n_test = max(1, int(round(idx.numel() * test_frac)))
        if idx.numel() <= 1:
            n_test = 0
        test.append(perm[:n_test])
        train.append(perm[n_test:])
    return torch.cat(train), torch.cat(test)


def probe(X: torch.Tensor, y: torch.Tensor, num_classes: int, seed: int = 0):
    tr, te = stratified_split(y, seed=seed)
    Xtr, Xte = X[tr].float(), X[te].float()
    ytr, yte = y[tr].long(), y[te].long()
    mean = Xtr.mean(dim=0)
    std = Xtr.std(dim=0).clamp_min(1e-6)
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    model = nn.Linear(X.shape[1], num_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    counts = torch.bincount(ytr, minlength=num_classes).float()
    weights = (ytr.numel() / (num_classes * counts.clamp_min(1.0)))
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr), ytr, weight=weights)
        loss.backward()
        opt.step()
    pred = model(Xte).argmax(dim=1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(yte.tolist(), pred.tolist()):
        cm[t, p] += 1
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    prec = tp / (tp + fp).clamp_min(1.0)
    rec = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
    support = cm.sum(dim=1) > 0
    return {
        "balanced_acc": float(rec[support].mean()) if support.any() else 0.0,
        "macro_f1": float(f1[support].mean()) if support.any() else 0.0,
        "n_train": int(ytr.numel()), "n_test": int(yte.numel()),
        "cm": cm.tolist(),
    }


def silhouette(X: torch.Tensor, y: torch.Tensor) -> float:
    if X.shape[0] > 2500:
        idx = torch.randperm(X.shape[0])[:2500]
        X, y = X[idx], y[idx]
    Z = X.float()
    Z = (Z - Z.mean(0)) / Z.std(0).clamp_min(1e-6)
    d = torch.cdist(Z, Z)
    vals = []
    classes = y.unique()
    for c in classes:
        same = y == c
        if same.sum() <= 1:
            continue
        a = d[same][:, same].sum(1) / (same.sum() - 1)
        b = None
        for other in classes:
            if other == c:
                continue
            mask = y == other
            if mask.any():
                bm = d[same][:, mask].mean(1)
                b = bm if b is None else torch.minimum(b, bm)
        if b is not None:
            vals.extend(((b - a) / torch.maximum(a, b).clamp_min(1e-9)).tolist())
    return float(sum(vals) / len(vals)) if vals else 0.0


def cosine_gap(X: torch.Tensor, y: torch.Tensor):
    Z = F.normalize(X.float(), dim=1)
    cos = Z @ Z.t()
    same = y[:, None] == y[None, :]
    tri = torch.triu(torch.ones_like(same), diagonal=1)
    sv = cos[tri & same]
    dv = cos[tri & ~same]
    s = float(sv.mean()) if sv.numel() else 0.0
    d = float(dv.mean()) if dv.numel() else 0.0
    return s, d, s - d


def kmeans(X: torch.Tensor, k: int, seed: int = 0, iters: int = 100):
    g = torch.Generator().manual_seed(seed)
    centers = X[torch.randperm(X.shape[0], generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(X, centers).argmin(dim=1)
        new = torch.stack([
            X[assign == j].mean(0) if (assign == j).any() else centers[j]
            for j in range(k)])
        if torch.allclose(new, centers):
            break
        centers = new
    return assign, centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--out", type=Path, default=ROOT / "audit/cnn_domain_audit.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = compose_cfg(args.batch)
    feats, labels, task_names, policy = collect(cfg, args.batches, device)
    print("collected", {k: tuple(v.shape) for k, v in feats.items()},
          "labels", {k: tuple(v.shape) for k, v in labels.items()})

    style = labels["style"]
    layout = labels["layout"]
    task = labels["task"]
    style_min = int(style.min())
    # Current 3 contiguous groups (20 styles each).
    style3 = ((style - style_min) // 20).clamp(0, 2)
    layout10 = ((layout - int(layout.min())) // 5).clamp(0, 9)
    num_tasks = int(task.max()) + 1
    task10 = torch.div(task * 10, max(num_tasks, 1), rounding_mode="floor").clamp(0, 9)

    results = []
    best = {}
    for (view, layer), X in sorted(feats.items()):
        base = {"view": view, "layer": layer}
        for name, y, k in [
            ("style_contiguous3", style3, 3),
            ("layout_coarse10", layout10, 10),
            ("task_coarse10", task10, 10),
            ("style_exact", style - style_min, int((style - style_min).max()) + 1),
        ]:
            m = probe(X, y - y.min(), int(y.max() - y.min()) + 1, seed=layer)
            s, d, gap = cosine_gap(X, y)
            sil = silhouette(X, y)
            row = {**base, "label": name, **m, "silhouette": sil,
                   "same_cos": s, "diff_cos": d, "cos_gap": gap}
            results.append(row)
            if name not in best or row["balanced_acc"] > best[name]["balanced_acc"]:
                best[name] = row

        # Style KMeans clusters on per-style means from the training split.
        tr, _ = stratified_split(style, seed=layer)
        Xtr = X[tr]
        ytr = style[tr] - style_min
        style_ids = ytr.unique()
        means = torch.stack([Xtr[ytr == sid].mean(0) for sid in style_ids])
        means = (means - means.mean(0)) / means.std(0).clamp_min(1e-6)
        for k in (3, 5, 10):
            if style_ids.numel() < k:
                continue
            assign, _ = kmeans(means, k, seed=layer)
            style_to_cluster = {int(sid): int(c) for sid, c in zip(style_ids, assign)}
            yclu = torch.tensor([style_to_cluster[int(s)] for s in (style - style_min)],
                                dtype=torch.long)
            m = probe(X, yclu, k, seed=layer)
            s, d, gap = cosine_gap(X, yclu)
            sil = silhouette(X, yclu)
            row = {**base, "label": f"style_kmeans{k}", **m,
                   "silhouette": sil, "same_cos": s, "diff_cos": d,
                   "cos_gap": gap,
                   "clusters": {int(c): [int(style_ids[i]) for i in range(style_ids.numel())
                                          if int(assign[i]) == int(c)]
                                for c in assign.unique()}}
            results.append(row)
            if row["label"] not in best or row["balanced_acc"] > best[row["label"]]["balanced_acc"]:
                best[row["label"]] = row

        # Layout-residualized style probe.
        tr, _ = stratified_split(style, seed=layer + 1)
        layout_mean = {}
        for lid in layout[tr].unique():
            mask = layout[tr] == lid
            layout_mean[int(lid)] = X[tr][mask].mean(0)
        global_mean = X[tr].mean(0)
        Xres = X.clone()
        for i in range(X.shape[0]):
            Xres[i] = X[i] - layout_mean.get(int(layout[i]), global_mean)
        m = probe(Xres, style3, 3, seed=layer + 2)
        s, d, gap = cosine_gap(Xres, style3)
        sil = silhouette(Xres, style3)
        row = {**base, "label": "style_contiguous3_layout_resid", **m,
               "silhouette": sil, "same_cos": s, "diff_cos": d, "cos_gap": gap}
        results.append(row)
        if row["label"] not in best or row["balanced_acc"] > best[row["label"]]["balanced_acc"]:
            best[row["label"]] = row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"batches": args.batches, "batch_size": args.batch,
               "n_samples": int(style.numel()),
               "task_names": task_names, "best": best, "all": results}
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    print("\nBEST PER LABEL")
    for name in sorted(best):
        r = best[name]
        print(f"{name:32s} view={r['view']:<24s} layer={r['layer']} "
              f"bal_acc={r['balanced_acc']:.3f} f1={r['macro_f1']:.3f} "
              f"sil={r['silhouette']:+.3f} cos_gap={r['cos_gap']:+.3f}")


if __name__ == "__main__":
    main()
