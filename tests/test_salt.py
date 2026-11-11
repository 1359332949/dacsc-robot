"""Phase-1 acceptance tests for the SALT port to CoRP (no SG-SALT).

Run:  python -m pytest tests/test_salt.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "CoRP"))

from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.leftover_tta import sample_with_grad  # noqa: E402
from dacsc.salt_tta import (  # noqa: E402
    FeatureChangeGate,
    SaltKeyAdapter,
    SaltTTA,
    build_leftover_target,
    flow_interpolate,
    flow_matching_leftover_loss,
)

CKPT = ROOT / "repos" / "CoRP" / "checkpoints" / "corp_libero.pt"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def policy():
    p = load_policy(CKPT, device=DEV)
    return p


@pytest.fixture(scope="module")
def batch(policy):
    rng = np.random.default_rng(0)
    views = policy.config["view_names"]
    images = {v: torch.from_numpy(rng.integers(0, 255, (1, 224, 224, 3), dtype=np.uint8)) for v in views}
    state = torch.from_numpy(rng.normal(size=(1, policy.config["state_dim"])).astype(np.float32))
    return prepare_observation(
        {"images": images, "state": state, "instruction": ["do it"]},
        config=policy.config, normalizer=policy.normalizer, device=torch.device(DEV),
    )


@pytest.fixture()
def adapter(policy):
    a = SaltKeyAdapter(policy.action_generator.blocks[0].attention.qkv, policy.config["hidden_dim"])
    yield a
    a.close()


def test_qkv_split_order(policy):
    att = policy.action_generator.blocks[0].attention
    x = torch.randn(1, 3, policy.config["hidden_dim"], device=DEV)
    q, k, v = att.project_qkv(x)
    w = att.qkv.weight
    b = att.qkv.bias
    d = policy.config["hidden_dim"]
    heads, hd = 12, d // 12
    q_lin = F.linear(x, w[:d], b[:d]).view(1, 3, heads, hd).permute(0, 2, 1, 3)
    k_lin = F.linear(x, w[d:2*d], b[d:2*d]).view(1, 3, heads, hd).permute(0, 2, 1, 3)
    v_lin = F.linear(x, w[2*d:], b[2*d:]).view(1, 3, heads, hd).permute(0, 2, 1, 3)
    # project_qkv: (B,T,3,heads,hd) -> Q/K/V, each (B,heads,T,hd)
    assert torch.allclose(q, q_lin, atol=1e-4)
    assert torch.allclose(k, k_lin, atol=1e-4)
    assert torch.allclose(v, v_lin, atol=1e-4)


def test_only_key_is_trainable(policy):
    tta = SaltTTA(policy)
    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    assert trainable == [], "all CoRP parameters must be frozen"
    assert tta.adapter.beta.requires_grad
    tta.close()


def test_q_v_output_unchanged(adapter, policy):
    w = adapter.effective_qkv_weight()
    d = policy.config["hidden_dim"]
    assert torch.equal(w[:d], adapter.w_qkv0[:d])
    assert torch.equal(w[2*d:], adapter.w_qkv0[2*d:])
    assert torch.equal(policy.action_generator.blocks[0].attention.projection.weight,
                       policy.action_generator.blocks[0].attention.projection.weight)


def test_zero_beta_is_exact_identity(adapter):
    d = adapter.hidden_dim
    w0 = adapter.w_qkv0[d:2*d]
    assert torch.equal(adapter.effective_key_weight(), w0)


def test_beta_bound_on_singular_values(adapter):
    with torch.no_grad():
        adapter.beta.fill_(1.0)   # beyond eta
    lo, hi = adapter.singular_value_ratio()
    import math
    assert lo >= math.exp(-adapter.eta) - 1e-3
    assert hi <= math.exp(adapter.eta) + 1e-3


def test_sample_with_grad_matches_sample(policy, batch):
    rep = policy.representation_extractor(**batch)
    torch.manual_seed(7)
    with torch.no_grad():
        ref = policy.action_generator.sample(rep)
    torch.manual_seed(7)
    out = sample_with_grad(policy.action_generator, rep, policy.config["action_generator"]["num_sample_steps"])
    assert torch.equal(ref, out)


def test_normalize_unnormalize_gripper(policy):
    a = torch.randn(4, 7, device=DEV)
    u = policy.normalizer.unnormalize_action(a)
    low = policy.normalizer.action_min.to(DEV)
    high = policy.normalizer.action_max.to(DEV)
    expect6 = 0.5 * (a[:, :6] + 1.0) * (high[:6] - low[:6]) + low[:6]
    assert torch.allclose(u[:, :6], expect6, atol=1e-5)
    assert torch.equal(u[:, 6], a[:, 6])          # gripper untouched


def test_leftover_alignment():
    h, a, e = 12, 7, 6
    target = torch.arange(h * a, dtype=torch.float32).reshape(h, a)
    y, mask = build_leftover_target(target, e, h, a)
    assert torch.equal(y[: h - e], target[e:])
    assert torch.equal(y[h - e :], torch.zeros_like(y[h - e :]))
    assert mask[: h - e].all() and mask[h - e :].eq(0).all()


def test_flow_direction():
    y = torch.randn(12, 7)
    eps = torch.randn(12, 7)
    tau = torch.tensor(0.25)
    x, vt = flow_interpolate(y, eps, tau)
    assert torch.allclose(x, 0.75 * y + 0.25 * eps)
    assert torch.allclose(vt, eps - y)


def test_loss_mask_only_overlap(policy, batch):
    rep = policy.representation_extractor(**batch)
    target = torch.zeros(12, 7, device=DEV)
    torch.manual_seed(3)
    loss = flow_matching_leftover_loss(policy.action_generator, rep, target, exec_len=6, draws=2)
    assert torch.isfinite(loss)
    target2 = target.clone(); target2[:6] = 100.0     # already-executed part must not matter
    torch.manual_seed(3)
    loss2 = flow_matching_leftover_loss(policy.action_generator, rep, target2, exec_len=6, draws=2)
    assert torch.allclose(loss, loss2)


def test_gradients_only_beta(policy, batch):
    tta = SaltTTA(policy)
    rep = policy.representation_extractor(**batch)
    target = torch.zeros(12, 7, device=DEV)
    torch.manual_seed(1)
    loss = flow_matching_leftover_loss(policy.action_generator, rep, target, exec_len=6, draws=1)
    tta.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert tta.adapter.beta.grad is not None and float(tta.adapter.beta.grad.norm()) > 0
    assert all(p.grad is None for p in policy.parameters())
    tta.close()


def test_prefix_recomputed_after_update(policy, batch):
    tta = SaltTTA(policy)
    rep = policy.representation_extractor(**batch)
    p0 = policy.action_generator.prepare_prefix(rep)[0].key.detach().clone()
    with torch.no_grad():
        tta.adapter.beta.fill_(0.1)
    p1 = policy.action_generator.prepare_prefix(rep)[0].key.detach().clone()
    assert not torch.equal(p0, p1)
    tta.close()


def test_gate_calibration_trigger_latch():
    gate = FeatureChangeGate(z=3.0)
    torch.manual_seed(0)
    base = torch.randn(1, 60, 768)
    changes = []
    prev = None
    for step in range(30):
        tok = base + 0.01 * torch.randn_like(base)
        c = gate.token_change(tok)
        if step > 0:
            changes.append(c)
    gate.calibrate(changes)
    gate.reset()
    gate.prev = None
    gate.update(base)
    assert not gate.active
    shock = base.clone(); shock[:, :48] = -base[:, :48]      # large agentview change
    triggered = gate.update(shock)
    assert triggered
    assert gate.update(base)                                  # latched


def test_episode_reset(policy):
    tta = SaltTTA(policy)
    with torch.no_grad():
        tta.adapter.beta.fill_(0.05)
    old = tta.optimizer
    tta.reset()
    assert torch.equal(tta.adapter.beta, torch.zeros_like(tta.adapter.beta))
    assert tta.optimizer is not old
    tta.close()


def test_frozen_identity(policy, batch):
    def run():
        torch.manual_seed(11)
        with torch.no_grad():
            return policy.predict(batch)

    ref = run()
    tta = SaltTTA(policy)
    out = run()
    tta.close()
    assert torch.equal(ref, out)
