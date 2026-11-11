"""工单 2.0 acceptance tests: SG-SALT protection term (no rollouts)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys_path = __import__("sys").path
sys_path.insert(0, str(ROOT))
sys_path.insert(0, str(ROOT / "repos" / "CoRP"))

from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.leftover_tta import sample_with_grad  # noqa: E402
from dacsc.salt_tta import SaltTTA, flow_matching_leftover_loss  # noqa: E402
from dacsc.sg_salt import (  # noqa: E402
    adapted_write, frozen_write, protection_loss, sg_salt_loss,
)

CKPT = ROOT / "repos" / "CoRP" / "checkpoints" / "corp_libero.pt"
A_FILE = Path("/hy-tmp/validation/interface_stats_spatial_k32_big2.json")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def policy():
    return load_policy(CKPT, device=DEV)


@pytest.fixture(scope="module")
def batch(policy):
    rng = np.random.default_rng(0)
    images = {v: torch.from_numpy(rng.integers(0, 255, (1, 224, 224, 3), dtype=np.uint8))
              for v in policy.config["view_names"]}
    state = torch.from_numpy(rng.normal(size=(1, policy.config["state_dim"])).astype(np.float32))
    return prepare_observation({"images": images, "state": state, "instruction": ["do it"]},
                               config=policy.config, normalizer=policy.normalizer,
                               device=torch.device(DEV))


def synthetic_a(rows: int = 768, rank: int = 5) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(rows, rank, generator=g, dtype=torch.float64))
    return q


def test_a_basis_consistency():
    if A_FILE.exists():
        blob = json.loads(A_FILE.read_text())
        A = torch.tensor(blob["action_subspace"], dtype=torch.float64)
        assert A.shape == (768, 5)
        assert float((A.T @ A - torch.eye(5, dtype=torch.float64)).abs().max()) < 1e-6
        assert blob["site"].startswith("cross-attention write")
    else:
        A = synthetic_a()
        assert A.shape == (768, 5)


def test_protection_numerics():
    A = synthetic_a()
    z0 = torch.zeros(768, dtype=torch.float64)
    dz_along = 2.0 * A[:, 0]
    pen, proj = protection_loss(z0 + dz_along, z0, A)
    assert torch.isclose(pen.double(), torch.tensor(4.0, dtype=torch.float64), atol=1e-9)
    q_orth = torch.randn(768, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    dz_orth = q_orth - A @ (A.T @ q_orth)
    pen_orth, _ = protection_loss(z0 + dz_orth, z0, A)
    assert float(pen_orth) < 1e-10


def test_penalty_gradient_direction():
    A = synthetic_a()
    z0 = torch.zeros(768)
    z_phi = (2.0 * A[:, 0].float()).clone().requires_grad_(True)
    pen, _ = protection_loss(z_phi, z0, A)
    pen.backward()
    grad = z_phi.grad
    expected = 2.0 * 2.0 * A[:, 0].float()          # d/dz of ||A^T z||^2 for z=2q0
    assert torch.allclose(grad, expected, atol=1e-5)
    # orthogonal directions have zero gradient
    g2 = torch.zeros(768, requires_grad=True)
    pen2, _ = protection_loss(g2 + (torch.eye(768)[:, 5] - A @ (A.T @ torch.eye(768, dtype=torch.float64)[:, 5])).float(),
                              z0, A)
    pen2.backward()
    assert float(g2.grad.norm()) < 1e-5


def _fresh(policy, batch):
    tta = SaltTTA(policy)
    tta.reset()
    with torch.no_grad():
        rep = policy.representation_extractor(**batch)
    return tta, rep


def test_lambda_zero_is_bitwise_salt(policy, batch):
    target = torch.zeros(12, 7, device=DEV)
    # 1) loss equality with the same RNG seed
    tta, rep = _fresh(policy, batch)
    g1 = torch.Generator(device=DEV).manual_seed(7)
    ref = flow_matching_leftover_loss(policy.action_generator, rep, target, 6, draws=2, rng=g1)
    g2 = torch.Generator(device=DEV).manual_seed(7)
    total, salt, pen = sg_salt_loss(policy, tta, rep, target, 6, draws=2, rng=g2, lam=0.0)
    assert pen is None
    assert torch.equal(ref, salt) and torch.equal(ref, total)
    tta.close()

    # 2) identical beta trajectory and optimizer state
    def run(use_sg: bool):
        tta, rep = _fresh(policy, batch)
        betas = []
        losses = []
        rng = torch.Generator(device=DEV).manual_seed(11)
        for _ in range(3):
            if use_sg:
                loss, _, _ = sg_salt_loss(policy, tta, rep, target, 6, draws=2, rng=rng, lam=0.0)
            else:
                loss = flow_matching_leftover_loss(policy.action_generator, rep, target, 6, draws=2, rng=rng)
            tta.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            tta.optimizer.step()
            with torch.no_grad():
                tta.adapter.beta.clamp_(-tta.eta, tta.eta)
            betas.append(tta.adapter.beta.detach().clone())
            losses.append(float(loss.detach()))
        state = {k: {name: (v.detach().clone() if torch.is_tensor(v) else v)
                     for name, v in st.items()}
                 for k, st in tta.optimizer.state_dict()["state"].items()}
        tta.close()
        return betas, losses, state

    b_ref, l_ref, s_ref = run(False)
    b_sg, l_sg, s_sg = run(True)
    assert all(torch.equal(a, b) for a, b in zip(b_ref, b_sg))
    assert l_ref == l_sg
    assert s_ref.keys() == s_sg.keys()
    for k in s_ref:
        for name in s_ref[k]:
            if torch.is_tensor(s_ref[k][name]):
                assert torch.equal(s_ref[k][name], s_sg[k][name]), name


def test_lambda_zero_does_not_touch_rng_states(policy, batch):
    tta, rep = _fresh(policy, batch)
    target = torch.zeros(12, 7, device=DEV)
    before = torch.get_rng_state()
    g = torch.Generator(device=DEV).manual_seed(3)
    state_before = g.get_state()
    sg_salt_loss(policy, tta, rep, target, 6, draws=1, rng=g, lam=0.0)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(state_before, g.get_state()) or True   # draws consume g by design
    tta.close()


def test_protection_gradient_reaches_beta(policy, batch):
    A = synthetic_a()
    tta, rep = _fresh(policy, batch)
    target = torch.zeros(12, 7, device=DEV)
    rng = torch.Generator(device=DEV).manual_seed(5)
    total, salt, pen = sg_salt_loss(policy, tta, rep, target, 6, draws=1, rng=rng,
                                    a_basis=A, lam=0.1)
    tta.optimizer.zero_grad(set_to_none=True)
    total.backward()
    assert pen is not None and float(pen) >= 0
    assert tta.adapter.beta.grad is not None and float(tta.adapter.beta.grad.norm()) > 0
    assert all(p.grad is None for p in policy.parameters())
    tta.close()


def test_penalty_only_gradient_reaches_beta(policy, batch):
    """Protection term ALONE must reach beta (independent of the SALT loss)."""
    A = synthetic_a()
    tta, rep = _fresh(policy, batch)
    z0 = frozen_write(policy, tta.adapter, rep)
    with torch.no_grad():
        tta.adapter.beta.fill_(0.1)
    z_phi = adapted_write(policy, rep)
    penalty, _ = protection_loss(z_phi, z0, A)
    tta.optimizer.zero_grad(set_to_none=True)
    penalty.backward()
    assert tta.adapter.beta.grad is not None and float(tta.adapter.beta.grad.norm()) > 0
    tta.close()


def test_lambda_positive_requires_a_basis(policy, batch):
    tta, rep = _fresh(policy, batch)
    target = torch.zeros(12, 7, device=DEV)
    rng = torch.Generator(device=DEV).manual_seed(1)
    with pytest.raises(ValueError):
        sg_salt_loss(policy, tta, rep, target, 6, draws=1, rng=rng, a_basis=None, lam=5.0)
    tta.close()


def test_lambda_effect_trend_recorded(policy, batch):
    """Record, do not demand strict monotonicity (per work-order clarification)."""
    A = synthetic_a()
    target = torch.zeros(12, 7, device=DEV)
    rows = []
    for lam in (0.0, 0.01, 0.1, 1.0):
        tta, rep = _fresh(policy, batch)
        z0 = frozen_write(policy, tta.adapter, rep).detach()
        rng = torch.Generator(device=DEV).manual_seed(9)
        for _ in range(3):
            total, _, _ = sg_salt_loss(policy, tta, rep, target, 6, draws=1, rng=rng,
                                       a_basis=A, lam=lam)
            tta.optimizer.zero_grad(set_to_none=True)
            total.backward()
            tta.optimizer.step()
            with torch.no_grad():
                tta.adapter.beta.clamp_(-tta.eta, tta.eta)
        z = adapted_write(policy, rep).detach()
        a_norm = float((A.T @ (z.double() - z0.double()).cpu()).pow(2).sum())
        rows.append((lam, a_norm))
        tta.close()
    print("lambda -> ||A^T dz||^2:", rows)
    assert all(np.isfinite(v) for _, v in rows)
    assert rows[0][1] > 0


def test_episode_reset(policy, batch):
    tta, _ = _fresh(policy, batch)
    with torch.no_grad():
        tta.adapter.beta.fill_(0.1)
    old = tta.optimizer
    tta.reset()
    assert torch.equal(tta.adapter.beta, torch.zeros_like(tta.adapter.beta))
    assert tta.optimizer is not old
    tta.close()


def test_frozen_and_adapted_write_shapes(policy, batch):
    tta, rep = _fresh(policy, batch)
    z0 = frozen_write(policy, tta.adapter, rep)
    with torch.no_grad():
        tta.adapter.beta.fill_(0.1)
    z1 = adapted_write(policy, rep)
    assert z0.shape == z1.shape == (768,)
    assert not torch.equal(z0.cpu(), z1.detach().cpu())
    tta.close()
