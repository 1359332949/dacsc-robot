"""工单 4.1 acceptance tests for SG-CPA (local implementation only)."""

from __future__ import annotations

import json
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
from dacsc.sg_cpa import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, SGCpa, from_pixel, noise_views, to_pixel,
)

CKPT = ROOT / "repos" / "CoRP" / "checkpoints" / "corp_libero.pt"
A_FILE = Path("/hy-tmp/validation/interface_stats_spatial_k32_big2.json")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def policy():
    return load_policy(CKPT, device=DEV)


@pytest.fixture(scope="module")
def prepared(policy):
    rng = np.random.default_rng(0)
    images = {v: torch.from_numpy(rng.integers(0, 255, (1, 224, 224, 3), dtype=np.uint8))
              for v in policy.config["view_names"]}
    state = torch.from_numpy(rng.normal(size=(1, policy.config["state_dim"])).astype(np.float32))
    return prepare_observation({"images": images, "state": state, "instruction": ["do it"]},
                               config=policy.config, normalizer=policy.normalizer,
                               device=torch.device(DEV))


@pytest.fixture(scope="module")
def a_basis():
    if not A_FILE.exists():
        pytest.skip("real A basis not available")
    return torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float64)


@pytest.fixture()
def cpa(policy, a_basis):
    obj = SGCpa(policy, a_basis, alpha=0.1, sigma=0.05, eta=0.05, seed=1234)
    try:
        yield obj
    finally:
        obj.close()


# 1 ---------------------------------------------------------------- round-trip
def test_pixel_roundtrip():
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    pixel = torch.rand(1, 3, 8, 8) * 0.8 + 0.1
    norm = (pixel - mean) / std
    rt = from_pixel(to_pixel({"v": norm}))["v"]
    assert float((rt - norm).abs().max()) < 1e-5


# 2 --------------------------------------------------- deterministic noise views
def test_noise_views_deterministic_and_independent(prepared):
    names = list(prepared["images"])
    a = noise_views(prepared["images"], 0.05, [1, 2, 3], names)
    b = noise_views(prepared["images"], 0.05, [1, 2, 3], names)
    c = noise_views(prepared["images"], 0.05, [4, 5, 6], names)
    for x, y in zip(a, b):
        for n in names:
            assert torch.equal(x[n], y[n])
    for n in names:
        assert not torch.equal(a[0][n], c[0][n])
    # per-camera perturbations must differ within the same view
    assert not torch.equal(a[0][names[0]], a[0][names[1]])


# 3 ------------------------------------------------- teacher/student K=1 no-op
def test_teacher_student_noop(policy, prepared, cpa):
    seed = 777
    with torch.no_grad():
        rep = policy.representation_extractor(**prepared)
    torch.manual_seed(seed)
    with torch.no_grad():
        teacher = policy.action_generator.sample(rep)
    torch.manual_seed(seed)
    student = sample_with_grad(policy.action_generator, rep,
                               policy.config["action_generator"]["num_sample_steps"])
    assert torch.equal(student.detach(), teacher)
    loss = F.mse_loss(student, teacher.detach())
    cpa.adapter.beta.grad = None
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in grads)
    assert cpa.adapter.beta.grad is None or float(cpa.adapter.beta.grad.norm()) == 0.0
    assert float(loss.detach()) == 0.0


# 4 ------------------------------------------------------ anchored mean formula
def test_anchored_mean_formula(policy, prepared, a_basis):
    objs = [SGCpa(policy, a_basis, alpha=a, sigma=0.05, seed=1234) for a in (0.0, 0.1, 1.0)]
    try:
        infos = [o.build_target(prepared, seed=1234) for o in objs]
        a_id = infos[0]["a_id"]
        aug_mean = infos[0]["aug_mean"]
        assert torch.allclose(infos[0]["target"], a_id, atol=0.0, rtol=0.0)   # alpha = 0
        assert torch.allclose(infos[2]["target"], aug_mean, atol=1e-6, rtol=0)  # alpha = 1 (float rounding)
        expected = a_id + 0.1 * (aug_mean - a_id)
        assert torch.allclose(infos[1]["target"], expected, atol=1e-6)
    finally:
        for o in objs:
            o.close()


# 5 -------------------------------------------- distillation gradient correctness
def test_distill_gradient_correctness(policy, prepared, cpa):
    info = cpa.build_target(prepared, seed=1234)
    idx = 0
    base = cpa.loss(prepared, info, seed=1234)
    cpa.adapter.beta.grad = None
    base.backward()
    analytic = float(cpa.adapter.beta.grad[idx])
    eps = 1e-4
    with torch.no_grad():
        cpa.adapter.beta[idx] += eps
    lp = float(cpa.loss(prepared, info, seed=1234).detach())
    with torch.no_grad():
        cpa.adapter.beta[idx] -= 2 * eps
    lm = float(cpa.loss(prepared, info, seed=1234).detach())
    with torch.no_grad():
        cpa.adapter.beta[idx] += eps
    numeric = (lp - lm) / (2 * eps)
    assert abs(analytic - numeric) <= 0.05 * max(abs(analytic), abs(numeric), 1e-8) + 1e-6


# 6 -------------------------------------------------- real Jacobian + nullspace
def test_real_jacobian_and_nullspace(prepared, cpa):
    j = cpa.jacobian_a(prepared, seed=1234)
    assert j.shape == (5, 768)
    assert float(j.norm()) > 0
    p = cpa.nullspace(j)
    assert float((j @ p).norm() / (p.norm() + 1e-12)) < 1e-10


# 7 -------------------------------------------- projected SGD nullspace property
def test_projected_sgd_nullspace(prepared, cpa):
    j = cpa.jacobian_a(prepared, seed=1234)
    p = cpa.nullspace(j)
    g = torch.randn(j.shape[1], generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    d = p @ g
    assert float((j @ d).norm()) < 1e-9 * (d.norm() + 1e-12)


# 8 ----------------------------------------- finite-step A check and rollback path
def test_finite_step_rollback(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, alpha=0.1, sigma=0.05, eta=0.05, seed=1234,
                a_change_tol=1e-15, max_backtracks=2)
    try:
        rec = obj.step(prepared, seed=1234)
        assert rec["accepted"] is False
        assert float(obj.adapter.beta.detach().norm()) < 1e-9
    finally:
        obj.close()


# 9 ------------------------------------------------ reset and RNG pairing
def test_reset_and_rng_pairing(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, alpha=0.1, sigma=0.05, seed=1234)
    try:
        obj.step(prepared, seed=1234)
        assert float(obj.adapter.beta.detach().norm()) > 0
        t1 = obj.build_target(prepared, seed=1234)["target"]
        obj.reset(seed=1234)
        assert float(obj.adapter.beta.detach().norm()) == 0.0
        t2 = obj.build_target(prepared, seed=1234)["target"]
        assert torch.equal(t1, t2)
        t3 = obj.build_target(prepared, seed=999)["target"]
        assert not torch.equal(t1, t3)
    finally:
        obj.close()


# 11 ------------------------------------------------ explicit beta bound (P0-1)
def test_beta_bound_respected(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, alpha=0.1, sigma=0.05, eta=0.5, seed=1234,
                eta_param=0.02, a_change_tol=1e6)
    try:
        for _ in range(3):
            obj.step(prepared, seed=1234)
        assert float(obj.adapter.beta.detach().abs().max()) <= 0.02 + 1e-6
    finally:
        obj.close()


# 12 ------------------------------------------------------- RNG isolation (P0-3)
def test_rng_isolation(policy, prepared, cpa):
    before = torch.get_rng_state()
    cpa.write(prepared, seed=1234)
    assert torch.equal(before, torch.get_rng_state())
    with torch.no_grad():
        rep = policy.representation_extractor(**prepared)
    cpa._sample(rep, seed=1234, grad=False)
    assert torch.equal(before, torch.get_rng_state())


# 13 ------------------------------------- cumulative A reference per episode (P0-2)
def test_cumulative_reference(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, alpha=0.1, sigma=0.05, eta=0.2, seed=1234,
                a_change_tol=1e6)
    try:
        assert obj.z_ref is None
        obj.step(prepared, seed=1234)
        assert obj.z_ref is not None
        rec = obj.step(prepared, seed=1234)
        assert rec["cumulative_a_change"] is not None
        assert rec["local_a_change"] is not None
        # reference must equal the beta=0 write
        with torch.no_grad():
            saved = obj.adapter.beta.detach().clone()
            obj.adapter.beta.zero_()
        z_ref_fresh = obj.write(prepared, seed=1234).detach().double().cpu()
        with torch.no_grad():
            obj.adapter.beta.copy_(saved)
        assert float((obj.z_ref - z_ref_fresh).abs().max()) < 1e-6
    finally:
        obj.close()


# 14 ------------------------------------------ SVD nullspace properties (P0-b)
def test_svd_nullspace_properties(prepared, cpa):
    j = cpa.jacobian_a(prepared, seed=1234)
    p = cpa.nullspace(j)
    assert float((j @ p).norm() / (p.norm() + 1e-12)) < 1e-10
    assert float((p - p.T).abs().max()) < 1e-10
    assert float((p @ p - p).abs().max()) < 1e-8


# 10 --------------------------------------- frozen params and only-beta update
def test_only_beta_changes(policy, prepared, cpa):
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    rec = cpa.step(prepared, seed=1234)
    after = dict(policy.named_parameters())
    for n, p0 in before.items():
        assert torch.equal(p0, after[n].detach()), n
    assert all(not p.requires_grad for p in policy.parameters())
    assert rec["accepted"] or rec["skipped"]
    assert cpa.optimizer.param_groups[0]["params"] == [cpa.adapter.beta]
