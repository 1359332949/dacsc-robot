import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sg_b2_v3 import build_candidates, exec_beta, uniform_scale


def test_uniform_scale_direction_and_bound():
    torch.manual_seed(0)
    d = torch.randn(768, dtype=torch.float64)
    applied, alpha, inf = uniform_scale(d, bound=0.15)
    assert abs(float(applied.abs().max()) - 0.15) < 1e-6
    assert float(alpha) == 0.15 / inf
    cos = float((applied @ d) / (applied.norm() * d.norm()))
    assert cos > 0.999999


def test_uniform_scale_projection_preserved():
    torch.manual_seed(1)
    J = torch.randn(5, 768, dtype=torch.float64)
    # build a nullspace projector of J
    _, _, vh = torch.linalg.svd(J, full_matrices=False)
    P = torch.eye(768, dtype=torch.float64) - vh.T @ vh
    g = P @ torch.randn(768, dtype=torch.float64)
    applied, _, _ = uniform_scale(g, bound=0.15)
    assert float((J @ applied).norm()) < 1e-9
    assert float(applied.abs().max()) <= 0.15 + 1e-12


def test_exec_beta_float32_and_clamp():
    d = torch.tensor([0.2, -0.3, 0.1, -0.05], dtype=torch.float64)
    b = exec_beta(d)
    assert b.dtype == torch.float64
    assert float(b.abs().max()) <= 0.15 + 1e-12
    assert torch.equal(b, d.float().double().clamp(-0.15, 0.15))


def test_predicted_a_change_uses_exec_beta():
    torch.manual_seed(2)
    J = torch.randn(5, 768, dtype=torch.float64)
    applied = torch.randn(768, dtype=torch.float64) * 0.2
    b = exec_beta(applied)
    pred = (J @ b).norm()
    assert float(pred) >= 0.0
    assert abs(float(pred) - float((J @ applied.float().double().clamp(-0.15, 0.15)).norm())) < 1e-12


def test_build_candidates_smoke_no_shadowing():
    torch.manual_seed(3)
    J = torch.randn(5, 768, dtype=torch.float64)
    _, _, vh = torch.linalg.svd(J, full_matrices=False)
    P = torch.eye(768, dtype=torch.float64) - vh.T @ vh
    raw = {"proj": P @ torch.randn(768, dtype=torch.float64),
           "full": torch.randn(768, dtype=torch.float64),
           "sens": (torch.eye(768, dtype=torch.float64) - P) @ torch.randn(768, dtype=torch.float64)}
    prod = torch.randn(768, dtype=torch.float32) * 1e-5
    rec = {"raw_d_beta_norm": 1e-4, "d_beta_norm": 1e-5, "accepted": True,
           "backtracks": 0, "step_scale": 1.0, "local_a_change": 1e-8, "cumulative_a_change": 1e-8}
    cands, meta = build_candidates(raw, prod, rec, J, "a", "b", 0, 1, 0.05)
    assert set(cands) == {"raw_proj_100", "raw_full_100", "raw_sens_100", "prod_step_005"}
    for k, v in cands.items():
        assert float(v.abs().max()) <= 0.15 + 1e-6
        assert meta[k]["applied_norm"] > 0
    assert meta["raw_proj_100"]["predicted_a_change"] < 1e-6
    assert meta["prod_step_005"]["last_attempt_norm"] == 1e-5
