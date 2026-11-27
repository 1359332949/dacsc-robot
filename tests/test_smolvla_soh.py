"""Pure-tensor contract tests for SOH-VLA; no simulator or checkpoint needed."""
import torch

from dacsc.smolvla_soh import SOHModule, parameter_count


def _u_geo(dim=768, k=64):
    q, _ = torch.linalg.qr(torch.randn(dim, k))
    return q


def test_zero_scale_is_exact_identity():
    torch.manual_seed(0)
    m = SOHModule(_u_geo())
    x = torch.randn(1, 1024, 768, dtype=torch.bfloat16)
    fp = torch.randn(1, 768 * 5)
    y = m(x, fp)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert torch.equal(x, y), 'scale=0 must be bit-identical'


def test_projection_kills_geo_and_keeps_complement():
    torch.manual_seed(1)
    u = _u_geo().float()
    m = SOHModule(u)
    geo = u[:, :8].T[None].expand(1, 8, -1)
    dom = (torch.randn(1, 16, 768) @ (torch.eye(768) - u @ u.T)).float()
    assert torch.allclose(m.project_dom(geo), torch.zeros_like(geo), atol=1e-6)
    assert torch.allclose(m.project_dom(dom), dom, atol=1e-5)


def test_parameter_budget_and_shape():
    m = SOHModule(_u_geo())
    assert 1_000_000 < parameter_count(m) < 1_500_000
    x = torch.randn(2, 1024, 768, dtype=torch.bfloat16)
    fp = torch.randn(2, 768 * 5)
    with torch.no_grad():
        m.scale.fill_(1e-3)
    y = m(x, fp)
    assert y.shape == (2, 1024, 768) and y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all()


def test_fingerprint_shape_and_finiteness():
    torch.manual_seed(2)
    p3 = torch.randn(1, 1024, 768)
    p12 = torch.randn(1, 1024, 768)
    ref3 = (torch.rand(768) + 0.5, torch.rand(768) + 0.5)
    ref12 = (torch.rand(768) + 0.5, torch.rand(768) + 0.5)
    fp = SOHModule.build_raw_fingerprint(p3, p12, ref3, ref12)
    assert fp.shape == (1, 768 * 5)
    assert torch.isfinite(fp).all()
