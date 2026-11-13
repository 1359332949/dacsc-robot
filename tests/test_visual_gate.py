import torch
from dacsc.visual_gate import VisualTokenGate


def test_identity_gate_is_bitwise_noop():
    gate = VisualTokenGate(("agentview", "eye_in_hand"), 8, mode="identity")
    x = torch.randn(2, 4, 8)
    y = gate(x, "agentview")
    assert y is x
    assert torch.equal(y, x)


def test_scale_gate_is_per_view_and_dimension():
    gate = VisualTokenGate(("a", "b"), 4, mode="scale")
    with torch.no_grad():
        gate.gamma[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        gate.gamma[1] = torch.tensor([0.5, 0.5, 0.5, 0.5])
    xa = torch.ones(1, 2, 4); xb = torch.ones(1, 2, 4)
    ya = gate(xa, "a"); yb = gate(xb, "b")
    assert torch.allclose(ya, torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 4))
    assert torch.allclose(yb, torch.full((1, 2, 4), 0.5))


def test_unknown_view_raises():
    gate = VisualTokenGate(("a",), 4, mode="scale")
    try:
        gate(torch.ones(1, 2, 4), "missing")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
