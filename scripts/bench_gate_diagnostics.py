#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark the gate diagnostics hot path: old pair loop vs vectorized.

The old ``mask_iou_by_domain`` / ``score_separation_by_domain`` implementations
called ``.item()`` inside an O(N^2) Python loop.  With N=180 (batch 60 x To 2
x views 3, or batch 30 x 2 x 3) the four IoU calls plus one cosine call blocked
the GPU for ~19 s per training step.  This script measures the replacement and
checks numerical equivalence against the old loop.

Usage:
    python scripts/bench_gate_diagnostics.py --device cuda
    python scripts/bench_gate_diagnostics.py --device mps --sizes 30,60,180
    python scripts/bench_gate_diagnostics.py --device cpu --iters 1
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dacsc.hard_gate import (mask_iou_by_domain, score_separation_by_domain,  # noqa: E402
                             topk_ste_mask)


def old_iou(mask: torch.Tensor, did: torch.Tensor, drop: bool = False):
    keep = (mask <= 0) if drop else (mask > 0)
    same, diff = [], []
    for i in range(keep.shape[0]):
        for j in range(i + 1, keep.shape[0]):
            inter = (keep[i] & keep[j]).sum().item()
            union = (keep[i] | keep[j]).sum().item()
            if union == 0:
                continue
            v = inter / union
            (same if int(did[i]) == int(did[j]) else diff).append(v)
    s = sum(same) / len(same) if same else 0.0
    d = sum(diff) / len(diff) if diff else 0.0
    return s, d, s - d


def old_score_cos(scores: torch.Tensor, did: torch.Tensor):
    z = torch.nn.functional.normalize(scores.float(), dim=1)
    same, diff = [], []
    for i in range(z.shape[0]):
        for j in range(i + 1, z.shape[0]):
            v = float((z[i] * z[j]).sum())
            (same if int(did[i]) == int(did[j]) else diff).append(v)
    s = sum(same) / len(same) if same else 0.0
    d = sum(diff) / len(diff) if diff else 0.0
    return s, d, s - d


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(fn, device: torch.device, iters: int) -> float:
    fn()  # warm-up
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / max(iters, 1)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sizes", default="30,60,180")
    ap.add_argument("--channels", type=int, default=384)
    ap.add_argument("--domains", type=int, default=10)
    ap.add_argument("--iters", type=int, default=2)
    args = ap.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(0)
    print(f"device={device} channels={args.channels} domains={args.domains} "
          f"iters={args.iters}")
    print(f"{'N':>5} {'old 5-call block':>18} {'new 5-call block':>18} "
          f"{'speedup':>9}")
    for n in [int(x) for x in args.sizes.split(",") if x]:
        scores = torch.randn(n, args.channels, device=device)
        mask = topk_ste_mask(scores, drop_ratio=0.33, temperature=0.1)
        did = torch.arange(n, device=device) % args.domains

        # Numerical equivalence against the old loop.
        for drop in (False, True):
            got = mask_iou_by_domain(mask, did, drop=drop)
            ref = old_iou(mask, did, drop=drop)
            err = max(abs(a - b) for a, b in zip(got, ref))
            assert err < 1e-4, (n, drop, got, ref, err)
        got = score_separation_by_domain(scores, did)
        ref = old_score_cos(scores, did)
        err = max(abs(a - b) for a, b in zip(got, ref))
        assert err < 1e-4, (n, got, ref, err)

        def old_block():
            old_iou(mask, did)
            old_iou(mask, did)
            old_iou(mask, did, drop=True)
            old_iou(mask, did, drop=True)
            old_score_cos(scores, did)

        def new_block():
            mask_iou_by_domain(mask, did)
            mask_iou_by_domain(mask, did)
            mask_iou_by_domain(mask, did, drop=True)
            mask_iou_by_domain(mask, did, drop=True)
            score_separation_by_domain(scores, did)

        t_old = timed(old_block, device, args.iters)
        t_new = timed(new_block, device, args.iters)
        print(f"{n:>5} {t_old:>17.4f}s {t_new:>17.4f}s "
              f"{t_old / max(t_new, 1e-9):>8.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
