"""Train/evaluate the causal failure monitor with state-grouped splits."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.failure_monitor import CausalFailureMonitor


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def recall_at_fpr(y, p, fpr=0.05):
    y = np.asarray(y); p = np.asarray(p)
    neg = np.sort(p[y == 0])
    if len(neg) == 0: return float("nan")
    thr = np.quantile(neg, 1.0 - fpr)
    return float(np.mean(p[y == 1] >= thr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    groups = d["group"]; y_all = d["y"]
    ug = sorted(set(groups.tolist()))
    if len(ug) < 3: raise ValueError("need at least 3 state groups")
    group_pos = {g: bool(y_all[groups == g].max() > 0) for g in ug}
    if not any(group_pos.values()):
        raise ValueError("no failure groups; collect failed trajectories first")
    rng = np.random.default_rng(args.seed)
    pos_groups = [g for g in ug if group_pos[g]]
    neg_groups = [g for g in ug if not group_pos[g]]
    val_groups = set()
    if pos_groups: val_groups.add(pos_groups[0])
    if neg_groups: val_groups.add(neg_groups[0])
    rest = [g for g in ug if g not in val_groups]
    rng.shuffle(rest)
    target = max(2, len(ug)//5)
    val_groups.update(rest[:max(0, target - len(val_groups))])
    val = np.array([g in val_groups for g in groups])
    # Normalize with training statistics only; no target/val leakage.
    tr_mask = ~val
    stats = {}
    for key in ("state", "action", "visual_delta"):
        x = d[key].astype(np.float32)
        mu = x[tr_mask].mean(axis=(0, 1), keepdims=True)
        sd = x[tr_mask].std(axis=(0, 1), keepdims=True) + 1e-6
        stats[key] = (mu.squeeze(0), sd.squeeze(0))
    def norm(key):
        mu, sd = stats[key]; return ((d[key].astype(np.float32) - mu) / sd).astype(np.float32)
    def ds(mask):
        return TensorDataset(torch.from_numpy(norm("state")[mask]), torch.from_numpy(norm("action")[mask]),
                             torch.from_numpy(norm("visual_delta")[mask]), torch.from_numpy(d["y"][mask]))
    tr = DataLoader(ds(~val), batch_size=args.batch_size, shuffle=True)
    va = DataLoader(ds(val), batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalFailureMonitor(window_size=args.window, hidden_dim=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tr_y = d["y"][tr_mask]
    pos_weight = float((len(tr_y) - tr_y.sum()) / max(tr_y.sum(), 1.0))
    lossf = nn.BCELoss()
    for ep in range(args.epochs):
        model.train(); total = 0.0
        for s, a, v, y in tr:
            s, a, v, y = s.to(device), a.to(device), v.to(device), y.to(device)
            opt.zero_grad(); p = model(s, a, v)
            w = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.ones_like(y))
            loss = lossf(p, y) * w.mean(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(y)
        print(json.dumps({"epoch": ep, "train_loss": total / len(tr.dataset)}))
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for s, a, v, y in va:
            ps.append(model(s.to(device), a.to(device), v.to(device)).cpu().numpy()); ys.append(y.numpy())
    p = np.concatenate(ps); y = np.concatenate(ys)
    metrics = {"auc": auc(y, p), "recall_at_5pct_fpr": recall_at_fpr(y, p, 0.05),
               "n_val_windows": len(y), "val_positive_fraction": float(y.mean()),
               "val_has_both_classes": bool(len(set(y.tolist())) == 2)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "window": args.window, "hidden": args.hidden,
                "metrics": metrics, "val_groups": sorted(val_groups),
                "normalization": {k: (v[0].tolist(), v[1].tolist()) for k, v in stats.items()}}, args.out)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
