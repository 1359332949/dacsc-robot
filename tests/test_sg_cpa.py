"""SG-CPA acceptance tests (4.0 revised protocol: 4 views, medoid, actual-loss no-op)."""

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
    IMAGENET_MEAN, IMAGENET_STD, SGCpa, action_distance, from_pixel,
    group_distances, medoid_index, to_pixel,
)

CKPT = ROOT / "repos" / "CoRP" / "checkpoints" / "corp_libero.pt"
A_FILE = Path("/hy-tmp/validation/interface_stats_spatial_k32_big2.json")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
IDENTITY_VIEW = [("identity", lambda im, sd: im)]


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
    obj = SGCpa(policy, a_basis, seed=1234)
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


# 2 -------------------------------------------- four views deterministic/independent
def test_four_views(prepared, cpa):
    a = cpa.make_view_images(prepared)
    b = cpa.make_view_images(prepared)
    assert len(a) == 4
    for x, y in zip(a, b):
        for n in x:
            assert torch.equal(x[n], y[n])
    assert [n for n, _ in cpa.view_specs] == ["identity", "noise", "blur", "photo"]
    cam = next(iter(a[0]))
    for x in a[1:]:
        assert not torch.equal(a[0][cam], x[cam])


# 3 --------------------------------- actual loss path K=1 no-op (identity only)
def test_k1_actual_loss_noop(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, views=IDENTITY_VIEW, seed=1234)
    try:
        info = obj.build_target(prepared, seed=1234)
        assert info["medoid_index"] == 0
        assert torch.equal(info["target"], info["acts"][0])
        loss = obj.loss(prepared, info, seed=1234)
        obj.adapter.beta.grad = None
        loss.backward()
        assert float(loss.detach()) == 0.0
        assert obj.adapter.beta.grad is None or float(obj.adapter.beta.grad.norm()) == 0.0
    finally:
        obj.close()


# 4 --------------------------------------------------- medoid selection formula
def test_medoid_selection():
    actions = [torch.zeros(2, 7), torch.ones(2, 7), torch.full((2, 7), 2.0)]
    # pairwise sums: [1+2, 1+1, 2+1] -> index 1
    assert medoid_index(actions) == 1
    # with an outlier, the central point wins
    actions = [torch.zeros(2, 7), torch.full((2, 7), 0.1), torch.full((2, 7), 10.0)]
    assert medoid_index(actions) == 1


def test_medoid_target_is_a_real_action(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, seed=1234)
    try:
        info = obj.build_target(prepared, seed=1234)
        k = info["medoid_index"]
        assert 0 <= k < 4
        assert torch.equal(info["target"], info["acts"][k])
        assert len(info["view_distances"]) == 4
    finally:
        obj.close()


# 5 ------------------------------------------------------ anchored mean ablation
def test_anchored_mean_formula(policy, prepared, a_basis):
    objs = [SGCpa(policy, a_basis, aggregation="mean_anchored", alpha=a, seed=1234)
            for a in (0.0, 0.1, 1.0)]
    try:
        infos = [o.build_target(prepared, seed=1234) for o in objs]
        a_id = infos[0]["acts"][0]
        aug_mean = torch.stack(infos[0]["acts"][1:]).mean(dim=0)
        assert torch.allclose(infos[0]["target"], a_id, atol=0.0, rtol=0.0)
        assert torch.allclose(infos[2]["target"], aug_mean, atol=1e-6, rtol=0)
        expected = a_id + 0.1 * (aug_mean - a_id)
        assert torch.allclose(infos[1]["target"], expected, atol=1e-6)
    finally:
        for o in objs:
            o.close()


# 6 ------------------------------------------------ group distance (gripper native)
def test_group_distances():
    a = torch.zeros(3, 7)
    b = a.clone()
    b[:, :3] = 0.3
    b[:, 3:6] = 0.2
    b[:, 6] = 1.0
    g = group_distances(a, b)
    assert abs(g["xyz"] - 0.3) < 1e-6
    assert abs(g["rot"] - 0.2) < 1e-6
    assert abs(g["grip"] - 1.0) < 1e-6
    assert abs(g["all"] - (0.3 * 3 + 0.2 * 3 + 1.0) / 7) < 1e-6


# 7 -------------------------------------------- distillation gradient correctness
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


# 8 -------------------------------------------------- real Jacobian + nullspace
def test_real_jacobian_and_nullspace(prepared, cpa):
    j = cpa.jacobian_a(prepared, seed=1234)
    assert j.shape == (5, 768)
    assert float(j.norm()) > 0
    p = cpa.nullspace(j)
    assert float((j @ p).norm() / (p.norm() + 1e-12)) < 1e-10
    assert float((p - p.T).abs().max()) < 1e-10
    assert float((p @ p - p).abs().max()) < 1e-8


# 9 ------------------------------- Jacobian finite-difference independent check
def test_jacobian_finite_difference(prepared, cpa):
    j = cpa.jacobian_a(prepared, seed=1234)
    d = torch.randn(j.shape[1], generator=torch.Generator().manual_seed(0),
                    dtype=torch.float64)
    d = d / d.norm()
    vjp = j @ d
    errors = []
    for h in (1e-2, 1e-3):
        with torch.no_grad():
            cpa.adapter.beta.zero_()
        z0 = cpa.write(prepared, seed=1234).detach().double().cpu()
        with torch.no_grad():
            cpa.adapter.beta.copy_(h * d.float().to(DEV))
        zp = cpa.write(prepared, seed=1234).detach().double().cpu()
        with torch.no_grad():
            cpa.adapter.beta.copy_((-h) * d.float().to(DEV))
        zm = cpa.write(prepared, seed=1234).detach().double().cpu()
        with torch.no_grad():
            cpa.adapter.beta.zero_()
        fd = (cpa.a_basis.T @ (zp - zm)) / (2 * h)
        rel = float((fd - vjp).norm() / (vjp.norm() + 1e-12))
        errors.append(rel)
    assert errors[-1] < 0.05, errors


# 10 ------------------------------------------ projected SGD nullspace property
def test_projected_sgd_nullspace(prepared, cpa):
    j = cpa.jacobian_a(prepared, seed=1234)
    p = cpa.nullspace(j)
    g = torch.randn(j.shape[1], generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    d = p @ g
    assert float((j @ d).norm()) < 1e-9 * (d.norm() + 1e-12)


# 11 ---------------------------------------- finite-step check and rollback path
def test_finite_step_rollback(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, eta=0.05, seed=1234, a_change_tol=1e-15, max_backtracks=2)
    try:
        rec = obj.step(prepared, seed=1234)
        assert rec["accepted"] is False
        assert float(obj.adapter.beta.detach().norm()) < 1e-9
    finally:
        obj.close()


# 12 ------------------------------------------------ reset and RNG pairing
def test_reset_and_rng_pairing(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, seed=1234)
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


# 13 ------------------------------------------------ explicit beta bound
def test_beta_bound_respected(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, eta=0.5, eta_param=0.02, a_change_tol=1e6, seed=1234)
    try:
        for _ in range(3):
            obj.step(prepared, seed=1234)
        assert float(obj.adapter.beta.detach().abs().max()) <= 0.02 + 1e-6
    finally:
        obj.close()


# 14 ------------------------------------------------------- RNG isolation
def test_rng_isolation(policy, prepared, cpa):
    before = torch.get_rng_state()
    cpa.write(prepared, seed=1234)
    assert torch.equal(before, torch.get_rng_state())
    with torch.no_grad():
        rep = policy.representation_extractor(**prepared)
    cpa._sample(rep, seed=1234, grad=False)
    assert torch.equal(before, torch.get_rng_state())


# 15 --------------------------------------- frozen params and only-beta update
def test_only_beta_changes(policy, prepared, cpa):
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    rec = cpa.step(prepared, seed=1234)
    after = dict(policy.named_parameters())
    for n, p0 in before.items():
        assert torch.equal(p0, after[n].detach()), n
    assert all(not p.requires_grad for p in policy.parameters())
    assert rec["attempted"] is True
    assert cpa.optimizer.param_groups[0]["params"] == [cpa.adapter.beta]


# 16 ---------------------------------------- ablation without projection runs
def test_no_projection_ablation(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, use_projection=False, seed=1234, a_change_tol=1e6)
    try:
        rec = obj.step(prepared, seed=1234)
        assert rec["retained_fraction"] == 1.0
        assert rec["use_projection"] is False and rec["a_guard"] is True
    finally:
        obj.close()


# 17 ------------------------------- view noise is re-drawn for different replan seeds
def test_view_noise_per_replan(prepared, cpa):
    a = cpa.make_view_images(prepared, seed=100)
    b = cpa.make_view_images(prepared, seed=100)
    c = cpa.make_view_images(prepared, seed=200)
    cam = next(iter(a[0]))
    for x, y in zip(a, b):
        assert torch.equal(x[cam], y[cam])
    assert not torch.equal(a[1][cam], c[1][cam])      # noise view changes with seed


# 18 --------------------------- action latent is re-drawn for different replan seeds
def test_action_latent_per_replan(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, seed=1234)
    try:
        t1 = obj.build_target(prepared, seed=100, view_seed=101)["target"]
        t2 = obj.build_target(prepared, seed=200, view_seed=201)["target"]
        assert not torch.equal(t1, t2)
        # teacher and student share the same latent within one step
        info = obj.build_target(prepared, seed=100, view_seed=101)
        assert max(float((info["target"] - a).abs().mean()) for a in info["acts"]) < 0.5
    finally:
        obj.close()


# 18b ------------------- identity-only views, vary action seed only (latent test)
def test_action_latent_identity_only(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, views=IDENTITY_VIEW, seed=1234)
    try:
        t1 = obj.build_target(prepared, seed=100, view_seed=100)["target"]
        t2 = obj.build_target(prepared, seed=200, view_seed=100)["target"]
        assert not torch.equal(t1, t2)
    finally:
        obj.close()


# 19 ------------------------------ no-A-protection: no projection AND no A guard
def test_no_a_protection_guard_off(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, use_projection=False, a_guard=False,
                a_change_tol=1e-15, seed=1234)
    try:
        rec = obj.step(prepared, seed=1234)
        assert rec["accepted"] is True
        assert rec["cumulative_a_change"] is None
        assert rec["use_projection"] is False and rec["a_guard"] is False
    finally:
        obj.close()


# 20 --------------------------- no tangent projection but A guard still rejects
def test_no_tangent_projection_guard_on(policy, prepared, a_basis):
    obj = SGCpa(policy, a_basis, use_projection=False, a_guard=True,
                a_change_tol=1e-15, max_backtracks=1, seed=1234)
    try:
        rec = obj.step(prepared, seed=1234)
        assert rec["accepted"] is False
        assert rec["use_projection"] is False and rec["a_guard"] is True
    finally:
        obj.close()
