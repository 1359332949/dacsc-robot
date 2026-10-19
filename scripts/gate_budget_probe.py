#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门控预算项的快速 CPU 探针。

不训练视觉/动作网络，只优化 12×384 个门控标量，
回答“给定 u_disc / v_action 的统计量，budget 项能否在 500 步内追上 rho”。

用法：
  python3 scripts/gate_budget_probe.py --steps 500
  python3 scripts/gate_budget_probe.py --steps 500 --lambda-rho 10
  python3 scripts/gate_budget_probe.py --steps 500 --gate-lr 0.1
  python3 scripts/gate_budget_probe.py --steps 500 --only-budget
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from dacsc.losses import choose_rho, gate_regularizer


def run(args, lam_rho=None, only_budget=False, lr=None):
    torch.manual_seed(args.seed)
    L, C = args.layers, args.channels
    u = torch.randn(L, C)
    u = (u - u.mean(1, keepdim=True)) / u.std(1, keepdim=True).clamp_min(1e-6)
    v = torch.randn(L, C)
    v = (v - v.mean(1, keepdim=True)) / v.std(1, keepdim=True).clamp_min(1e-6)
    eta = torch.full((L, C), torch.logit(torch.tensor(args.gate_init)).item(),
                     requires_grad=True)
    opt = torch.optim.Adam([eta], lr=lr if lr is not None else args.gate_lr,
                           betas=(0.9, 0.95))
    rows = []
    for step in range(args.steps):
        g = torch.sigmoid(eta)
        rho = choose_rho(step, args.steps, args.rho_max)
        parts = gate_regularizer(
            g, u, v, rho=rho,
            lambda_d=0.0 if only_budget else args.lambda_d,
            lambda_a=0.0 if only_budget else args.lambda_a,
            lambda_rho=args.lambda_rho if lam_rho is None else lam_rho)
        opt.zero_grad()
        parts["loss"].backward()
        opt.step()
        if step in (0, 49, 99, 199, 299, 399, args.steps - 1):
            gg = torch.sigmoid(eta).detach()
            rows.append({
                "step": step + 1,
                "target_rho": rho,
                "suppression": float((1 - gg).mean()),
                "gate_mean": float(gg.mean()),
                "gate_min": float(gg.min()),
                "gate_max": float(gg.max()),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--channels", type=int, default=384)
    ap.add_argument("--gate-init", type=float, default=0.999)
    ap.add_argument("--gate-lr", type=float, default=1e-2)
    ap.add_argument("--lambda-rho", type=float, default=1.0)
    ap.add_argument("--lambda-d", type=float, default=1.0)
    ap.add_argument("--lambda-a", type=float, default=1.0)
    ap.add_argument("--rho-max", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-budget", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("audit/gate_budget_probe.json"))
    a = ap.parse_args()

    cases = {
        "current": run(a),
        "lambda_rho_10": run(a, lam_rho=10.0),
        "gate_lr_0.1": run(a, lr=0.1),
        "budget_only": run(a, only_budget=True),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"args": vars(a), "cases": cases}, indent=2,
                                ensure_ascii=False, default=str), encoding="utf-8")
    for name, rows in cases.items():
        last = rows[-1]
        print(f"{name:12s} suppression={last['suppression']:.4f} "
              f"target={last['target_rho']:.4f} gmean={last['gate_mean']:.5f} "
              f"gmin={last['gate_min']:.5f} gmax={last['gate_max']:.5f}")
    print("report:", a.out)


if __name__ == "__main__":
    main()
