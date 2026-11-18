"""Train a failure monitor with explicit source-fit/source-validation roles."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dacsc.failure_monitor import CausalFailureMonitor


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam): ctx.lam = lam; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return -ctx.lam * grad, None


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg): return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def recall_at_fpr(y, p, fpr=0.05):
    y = np.asarray(y); p = np.asarray(p)
    neg = np.sort(p[y == 0])
    if not len(neg): return float("nan")
    return float(np.mean(p[y == 1] >= np.quantile(neg, 1.0 - fpr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", type=Path, required=True)
    ap.add_argument("--val-data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--domain-adversarial", action="store_true")
    ap.add_argument("--domain-weight", type=float, default=0.1)
    args = ap.parse_args()
    tr_d = np.load(args.train_data, allow_pickle=True)
    va_d = np.load(args.val_data, allow_pickle=True)
    if len(set(tr_d["y"].tolist())) < 2:
        raise ValueError("source-fit training data has only one class")
    if len(set(va_d["y"].tolist())) < 2:
        raise ValueError("source-validation data has only one class")
    stats = {}
    for key in ("state", "action", "visual_delta"):
        x = tr_d[key].astype(np.float32)
        mu = x.mean(axis=(0, 1), keepdims=True); sd = x.std(axis=(0, 1), keepdims=True) + 1e-6
        stats[key] = (mu.squeeze(0), sd.squeeze(0))
    def prep(d):
        out = {}
        for key in ("state", "action", "visual_delta"):
            mu, sd = stats[key]
            out[key] = ((d[key].astype(np.float32) - mu) / sd).astype(np.float32)
        return out
    tr = prep(tr_d); va = prep(va_d)
    tr_dom = (tr_d["domain"] == "shift").astype(np.int64)
    def make(d, dom):
        return TensorDataset(torch.from_numpy(d["state"]), torch.from_numpy(d["action"]),
                             torch.from_numpy(d["visual_delta"]), torch.from_numpy(tr_d["y"] if d is tr else va_d["y"]),
                             torch.from_numpy(dom))
    train_loader = DataLoader(make(tr, tr_dom), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(make(va, (va_d["domain"] == "shift").astype(np.int64)), batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalFailureMonitor(window_size=args.window, hidden_dim=args.hidden,
                                 use_visual=not args.no_visual).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tr_y = tr_d["y"].astype(np.float32)
    pos_weight = float((len(tr_y) - tr_y.sum()) / max(tr_y.sum(), 1.0))
    lossf = nn.BCELoss()
    for ep in range(args.epochs):
        model.train(); total = 0.0
        for s, a, v, y, dom in train_loader:
            s, a, v, y, dom = s.to(device), a.to(device), v.to(device), y.to(device), dom.to(device)
            opt.zero_grad(); p = model(s, a, v)
            w = torch.where(y > 0.5, torch.full_like(y, pos_weight), torch.ones_like(y))
            loss = lossf(p, y) * w.mean()
            if args.domain_adversarial:
                h = model.encode(s, a, v)
                dlog = model.domain_head(_GradReverse.apply(h, args.domain_weight))
                loss = loss + args.domain_weight * nn.functional.cross_entropy(dlog, dom)
            loss.backward(); opt.step(); total += float(loss.detach()) * len(y)
        print(json.dumps({"epoch": ep, "train_loss": total / len(tr_d["y"])}))
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for s, a, v, y, _ in val_loader:
            ps.append(model(s.to(device), a.to(device), v.to(device)).cpu().numpy()); ys.append(y.numpy())
    p = np.concatenate(ps); y = np.concatenate(ys)
    metrics = {"auc": auc(y, p), "recall_at_5pct_fpr": recall_at_fpr(y, p, 0.05),
               "n_val_windows": len(y), "val_positive_fraction": float(y.mean()),
               "val_has_both_classes": bool(len(set(y.tolist())) == 2)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "window": args.window, "hidden": args.hidden,
                "metrics": metrics, "val_groups": sorted(set(va_d["group"].tolist())),
                "normalization": {k: (v[0].tolist(), v[1].tolist()) for k, v in stats.items()},
                "use_visual": not args.no_visual, "domain_adversarial": bool(args.domain_adversarial)}, args.out)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
