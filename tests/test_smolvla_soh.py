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


def test_u_geo_must_be_orthonormal():
    bad = torch.randn(768, 8)
    try:
        SOHModule(bad)
    except ValueError as e:
        assert 'orthonormal' in str(e)
    else:
        raise AssertionError('non-orthonormal u_geo accepted')


def test_fingerprint_views_are_not_overwritten():
    """Two camera views must produce two distinct [1, 3840] fingerprints."""
    torch.manual_seed(3)
    p3 = [torch.randn(1, 1024, 768), 2.0 + torch.randn(1, 1024, 768)]
    p12 = [torch.randn(1, 1024, 768), 3.0 + torch.randn(1, 1024, 768)]
    ref3 = (torch.rand(768) + 0.5, torch.rand(768) + 0.5)
    ref12 = (torch.rand(768) + 0.5, torch.rand(768) + 0.5)
    a = SOHModule.build_raw_fingerprint(p3[0], p12[0], ref3, ref12)
    b = SOHModule.build_raw_fingerprint(p3[1], p12[1], ref3, ref12)
    assert a.shape == b.shape == (1, 3840)
    assert not torch.allclose(a, b)


def test_streaming_ema_updates_per_view():
    from dacsc.smolvla_soh import SmolVLASOHAdapter
    a = object.__new__(SmolVLASOHAdapter)
    a.ema_beta = 0.8
    a.ema_fp = [None, None]
    x0 = torch.full((1, 3840), 1.0)
    x1 = torch.full((1, 3840), 5.0)
    assert torch.equal(a.update_ema(0, x0), x0)          # step-0 init
    assert torch.equal(a.update_ema(1, x1), x1)
    assert torch.allclose(a.update_ema(0, torch.zeros_like(x0)), 0.8 * x0)
    assert torch.allclose(a.ema_fp[1], x1)


def test_spectral_damping_is_smooth():
    torch.manual_seed(4)
    v, _ = torch.linalg.qr(torch.randn(768, 768))
    eig = torch.logspace(-4, 4, 768)
    m = SOHModule(None, v_basis=v, eigenvalues=eig, damping_beta=1.0)
    x = torch.randn(1, 16, 768)
    y = m.project_dom(x)
    # Most sensitive direction (last, large eigenvalue) is strongly damped.
    sensitive = v[:, -1][None, None, :]
    assert (sensitive @ m.v_basis).abs().max() > 0.99
    damped = m.project_dom(sensitive)
    assert damped.norm() < 0.05 * sensitive.norm()
    # Least sensitive direction is nearly preserved.
    weak = v[:, 0][None, None, :]
    kept = m.project_dom(weak)
    assert torch.allclose(kept, weak, atol=1e-2)
    assert torch.isfinite(y).all()
