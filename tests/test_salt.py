"""Phase-1.1 acceptance tests: CoRP-native flow direction + SALT B.4 gate."""

from __future__ import annotations

import sys
from pathlib import Path

import types

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
    _masked_mse,
    build_leftover_target,
    flow_interpolate,
    flow_matching_leftover_loss,
)

CKPT = ROOT / "repos" / "CoRP" / "checkpoints" / "corp_libero.pt"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def policy():
    return load_policy(CKPT, device=DEV)


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


class FakeConstantVelocity:
    """Minimal stand-in exposing the interface sample_with_grad needs."""

    def __init__(self, velocity: torch.Tensor, num_steps: int):
        self.v = velocity
        self.num_sample_steps = num_steps
        self.action_horizon = velocity.shape[0]
        self.action_dim = velocity.shape[1]

    def prepare_prefix(self, representation):
        return None

    def __call__(self, action, tau, prefix):
        return self.v.expand_as(action)


def test_corp_native_flow_integration_direction():
    """Constant v = Y-eps integrated from eps reproduces Y; reverse direction must fail."""
    torch.manual_seed(3)
    eps = torch.randn(12, 7)
    y = torch.randn(12, 7)
    rep = types.SimpleNamespace(tokens=torch.zeros(1, 1, 768), action_dtype=torch.float32)
    fake = FakeConstantVelocity(y - eps, num_steps=10)
    torch.manual_seed(3)
    out = sample_with_grad(fake, rep, num_steps=10)
    assert torch.allclose(out[0], y, atol=1e-5)
    # reverse convention (eps - y) does NOT integrate to y
    fake_bad = FakeConstantVelocity(eps - y, num_steps=10)
    torch.manual_seed(3)
    out_bad = sample_with_grad(fake_bad, rep, num_steps=10)
    assert not torch.allclose(out_bad[0], y, atol=1e-3)


def test_flow_interpolate_corp_convention():
    y = torch.randn(12, 7); eps = torch.randn(12, 7); tau = torch.tensor(0.25)
    x, vt = flow_interpolate(y, eps, tau)
    assert torch.allclose(x, 0.75 * eps + 0.25 * y)
    assert torch.allclose(vt, y - eps)


def test_qkv_split_order_and_bias(policy):
    att = policy.action_generator.blocks[0].attention
    x = torch.randn(1, 3, policy.config["hidden_dim"], device=DEV)
    q, k, v = att.project_qkv(x)
    w, b, d, heads = att.qkv.weight, att.qkv.bias, policy.config["hidden_dim"], 12
    hd = d // heads
    for got, sl, off in ((q, w[:d], 0), (k, w[d:2*d], d), (v, w[2*d:], 2*d)):
        want = F.linear(x, sl, b[off:off+d]).view(1, 3, heads, hd).permute(0, 2, 1, 3)
        assert torch.allclose(got, want, atol=1e-4)


def test_only_key_trainable(policy):
    tta = SaltTTA(policy)
    assert [n for n, p in policy.named_parameters() if p.requires_grad] == []
    assert tta.adapter.beta.requires_grad
    tta.close()


def test_q_v_output_and_bias_unchanged_after_update(policy, batch):
    att = policy.action_generator.blocks[0].attention
    proj_w0 = att.projection.weight.detach().clone()
    proj_b0 = att.projection.bias.detach().clone()
    tta = SaltTTA(policy)
    rep = policy.representation_extractor(**batch)
    target = torch.zeros(12, 7, device=DEV)
    for _ in range(2):
        loss = flow_matching_leftover_loss(policy.action_generator, rep, target, 6, draws=1, mode="self")
        tta.optimizer.zero_grad(set_to_none=True); loss.backward(); tta.optimizer.step()
    assert torch.equal(att.projection.weight, proj_w0)
    assert torch.equal(att.projection.bias, proj_b0)
    d = policy.config["hidden_dim"]
    w = tta.adapter.effective_qkv_weight()
    assert torch.equal(w[:d], tta.adapter.w_qkv0[:d])
    assert torch.equal(w[2*d:], tta.adapter.w_qkv0[2*d:])
    assert torch.equal(tta.adapter.qkv.bias, att.qkv.bias)
    tta.close()


def test_zero_beta_exact_identity(adapter):
    d = adapter.hidden_dim
    assert torch.equal(adapter.effective_key_weight(), adapter.w_qkv0[d:2*d])


def test_beta_bound(adapter):
    import math
    with torch.no_grad():
        adapter.beta.fill_(1.0)
    lo, hi = adapter.singular_value_ratio()
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
    low = policy.normalizer.action_min.to(DEV); high = policy.normalizer.action_max.to(DEV)
    assert torch.allclose(u[:, :6], 0.5 * (a[:, :6] + 1.0) * (high[:6] - low[:6]) + low[:6], atol=1e-5)
    assert torch.equal(u[:, 6], a[:, 6])


def test_leftover_alignment():
    h, a, e = 12, 7, 6
    target = torch.arange(h * a, dtype=torch.float32).reshape(h, a)
    y, mask = build_leftover_target(target, e, h, a)
    assert torch.equal(y[: h - e], target[e:])
    assert torch.equal(y[h - e :], torch.zeros_like(y[h - e :]))
    assert mask[: h - e].all() and mask[h - e :].eq(0).all()


def test_masked_mse_reduction():
    diff = torch.ones(12, 7)
    mask = torch.zeros(12, 1); mask[:6] = 1.0
    loss = _masked_mse(diff, mask)
    assert torch.isclose(loss, torch.tensor(1.0))     # mean over 6*7 supervised elements
    mask2 = torch.zeros(12, 1); mask2[:3] = 1.0
    assert torch.isclose(_masked_mse(diff, mask2), torch.tensor(1.0))
    assert torch.isclose(_masked_mse(torch.zeros(12, 7), mask2), torch.tensor(0.0))


def test_loss_ignores_non_overlap(policy, batch):
    rep = policy.representation_extractor(**batch)
    target = torch.zeros(12, 7, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(5)
    loss = flow_matching_leftover_loss(policy.action_generator, rep, target, 6, draws=2, rng=gen)
    target2 = target.clone(); target2[:6] = 100.0
    gen2 = torch.Generator(device=DEV).manual_seed(5)
    loss2 = flow_matching_leftover_loss(policy.action_generator, rep, target2, 6, draws=2, rng=gen2)
    assert torch.allclose(loss, loss2)


def test_gradients_only_beta(policy, batch):
    tta = SaltTTA(policy)
    rep = policy.representation_extractor(**batch)
    target = torch.zeros(12, 7, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(1)
    loss = flow_matching_leftover_loss(policy.action_generator, rep, target, 6, draws=1, rng=gen)
    tta.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert tta.adapter.beta.grad is not None and float(tta.adapter.beta.grad.norm()) > 0
    assert all(p.grad is None for p in policy.parameters())
    tta.close()


def test_prefix_recomputed_after_update(policy, batch):
    tta = SaltTTA(policy)
    rep = policy.representation_extractor(**batch)
    k0 = policy.action_generator.prepare_prefix(rep)[0].key.detach().clone()
    with torch.no_grad():
        tta.adapter.beta.fill_(0.1)
    k1 = policy.action_generator.prepare_prefix(rep)[0].key.detach().clone()
    assert not torch.equal(k0, k1)
    tta.close()


def test_beta_diagnostics():
    gate = FeatureChangeGate(num_tokens=96)
    qkv = torch.nn.Linear(768, 2304)
    adapter = SaltKeyAdapter(qkv, 768, eta=0.15)
    diag = adapter.diagnostics()
    assert diag["beta_l2"] == 0.0 and diag["frac_ge_095eta"] == 0.0
    with torch.no_grad():
        adapter.beta.fill_(0.15)
    diag = adapter.diagnostics()
    assert diag["frac_ge_099eta"] == 1.0
    assert abs(diag["beta_l2"] - 0.15 * (768 ** 0.5)) < 1e-3
    adapter.close()


def test_gate_b4_statistics_reference():
    torch.manual_seed(0)
    changes = [torch.rand(20, 96) * 0.01 for _ in range(30)]
    m, s = FeatureChangeGate.fit_stats(changes)
    mat = torch.stack(changes)
    med = mat.median(dim=0).values
    mad = 1.4826 * (mat - med).abs().median(dim=0).values
    floor = 0.25 * mad.median()
    want_s = mad.clamp_min(float(floor))
    assert torch.allclose(m, med)
    assert torch.allclose(s, want_s)
    gate = FeatureChangeGate(num_tokens=96)
    gate.median, gate.scale = m, s
    score = gate.score_raw(changes[0], m, s)
    z = ((changes[0] - m) / s).clamp(0.0, 10.0).mean()
    assert torch.isclose(score, z)


def test_gate_crossfit_threshold():
    torch.manual_seed(1)
    per_task = {}
    for tid in range(5):
        eps = []
        for _ in range(4):
            seq = [torch.rand(96) * 0.01 + 0.001 * tid for _ in range(10)]
            eps.append(seq)
        per_task[tid] = eps
    gate = FeatureChangeGate(num_tokens=96)
    stats = gate.calibrate(per_task)
    assert stats["threshold"] > 0
    assert gate.threshold == stats["threshold"]


def test_gate_requires_calibration():
    gate = FeatureChangeGate(num_tokens=96)
    gate.reset()
    with pytest.raises(RuntimeError):
        gate.update(torch.randn(1, 120, 768))


def test_gate_latch():
    gate = FeatureChangeGate(num_tokens=96)
    torch.manual_seed(0)
    base = torch.randn(1, 120, 768)
    seq = [base + 0.01 * torch.randn_like(base) for _ in range(30)]
    changes = []
    gate.reset()
    prev = None
    for tok in seq:
        c = gate.token_change(tok)
        if gate.prev is not None:
            changes.append(c)
    per_task = {0: [changes]}
    gate.calibrate(per_task)
    gate.reset()
    gate.update(base)
    shock = base.clone(); shock[:, :96] = -base[:, :96]
    assert gate.update(shock)
    assert gate.update(base)     # latched


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
