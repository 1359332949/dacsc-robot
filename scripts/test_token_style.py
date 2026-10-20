#!/usr/bin/env python3
"""Unit tests for token-level style-statistics modulation."""
from __future__ import annotations
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dacsc.token_style import TokenImportanceEMA, TokenStyleModulator  # noqa: E402


def main():
    torch.manual_seed(0)
    B, V, K, D = 4, 2, 8, 16
    tokens = torch.randn(B, V, K, D)
    did = torch.tensor([0, 0, 1, 1])

    # Identity in eval / mode=none.
    m0 = TokenStyleModulator(mode="none")
    assert torch.equal(m0(tokens, did, training=True), tokens)
    m0.mode = "random_stats"
    assert torch.equal(m0(tokens, did, training=False), tokens)
    print("① token style identity / eval: ✓")

    # random_stats with alpha=1 replaces each row's mean/std with a donor's.
    mr = TokenStyleModulator(mode="random_stats", alpha=1.0)
    out = mr(tokens, did, training=True)
    assert out.shape == tokens.shape and torch.isfinite(out).all()
    z = tokens.reshape(B * V, K, D)
    mu = z.mean(dim=1)
    sigma = z.std(dim=1, unbiased=False)
    donor = mr.last_donor
    assert donor is not None and donor.shape == (B * V,)
    assert torch.allclose(out.reshape(B * V, K, D).mean(dim=1), mu[donor], atol=2e-5)
    assert torch.allclose(out.reshape(B * V, K, D).std(dim=1, unbiased=False),
                          sigma[donor], atol=2e-5)
    print("② random statistics replacement: ✓")

    # cross_style_stats prefers donors from different domains.
    mc = TokenStyleModulator(mode="cross_style_stats", alpha=0.5)
    _ = mc(tokens, did, training=True)
    diff = float(mc.last_stats_t["diff_domain_frac"])
    assert diff > 0.8, diff
    print(f"③ cross-style donor fraction: ✓ ({diff:.3f})")

    # action-aware modulation leaves high-importance tokens almost unchanged.
    ma = TokenStyleModulator(mode="cross_style_stats_action_aware",
                             alpha=1.0, action_weight=1.0)
    ma.set_action_importance(torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    outa = ma(tokens, did, training=True)
    delta = (outa - tokens).abs().mean(dim=(0, 1, 3))
    assert delta[1].item() < 0.05 * delta[0].item(), delta
    assert delta[3].item() < 0.05 * delta[2].item(), delta
    print("④ action-aware token weighting: ✓", delta.tolist())

    # TokenImportanceEMA reads |token * grad| and normalizes to [0,1].
    t = torch.randn(6, K, D, requires_grad=True)
    (t * torch.arange(K, dtype=t.dtype)[None, :, None]).sum().backward()
    ema = TokenImportanceEMA(K, momentum=0.9)
    ema.update_from_tensor(t)
    assert ema.ready()
    n = ema.normalized()
    assert n.shape == (K,) and float(n.min()) >= 0.0 and float(n.max()) <= 1.0
    assert float(n[-1]) > float(n[0])
    print("⑤ token action importance EMA: ✓", ema.summary())

    print("token style tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
