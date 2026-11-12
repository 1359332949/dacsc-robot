import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sg_b2_v3 import uniform_scale


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
