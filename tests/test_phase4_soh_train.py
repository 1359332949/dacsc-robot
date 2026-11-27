"""Regression tests for the training-only bugs fixed after the first audit."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_train_module():
    try:
        import phase3_smolvla_common  # noqa: F401
    except Exception as exc:
        pytest.skip(f'SmolVLA runtime unavailable: {exc}')
    path = Path(__file__).resolve().parents[1] / 'scripts/phase4_soh_train.py'
    spec = importlib.util.spec_from_file_location('phase4_soh_train_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generic_perturb_cpu_generator():
    t = _load_train_module()
    x = torch.rand(1, 3, 32, 32)
    g = torch.Generator(device='cpu').manual_seed(0)
    for _ in range(8):
        y = t.generic_perturb(x, g)
        assert y.shape == x.shape and torch.isfinite(y).all()
        assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_generic_perturb_cuda_generator_matches_device():
    t = _load_train_module()
    x = torch.rand(1, 3, 32, 32, device='cuda')
    g = torch.Generator(device='cuda').manual_seed(0)
    for _ in range(8):
        y = t.generic_perturb(x, g)
        assert y.device == x.device and y.shape == x.shape and torch.isfinite(y).all()
