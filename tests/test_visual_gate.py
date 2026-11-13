import unittest

import torch

from dacsc.visual_gate import VisualTokenGate


class TestVisualGate(unittest.TestCase):
    def test_identity_is_bitwise_noop(self):
        gate = VisualTokenGate(("agentview", "eye_in_hand"), 8, mode="identity")
        x = torch.randn(2, 4, 8)
        y = gate(x, "agentview")
        self.assertIs(y, x)
        self.assertTrue(torch.equal(y, x))

    def test_identity_rejects_unknown_view(self):
        gate = VisualTokenGate(("a",), 4, mode="identity")
        with self.assertRaises(KeyError):
            gate(torch.ones(1, 2, 4), "missing")

    def test_identity_rejects_wrong_dim(self):
        gate = VisualTokenGate(("a",), 4, mode="identity")
        with self.assertRaises(ValueError):
            gate(torch.ones(1, 2, 5), "a")

    def test_suppression_is_per_view_and_bounded(self):
        gate = VisualTokenGate(("a", "b"), 4, mode="suppression")
        with torch.no_grad():
            gate.gamma[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
            gate.gamma[1] = torch.tensor([0.5, 0.5, 0.5, 0.5])
        ya = gate(torch.ones(1, 2, 4), "a")
        yb = gate(torch.ones(1, 2, 4), "b")
        self.assertTrue(torch.allclose(ya, torch.tensor([1.0, 1.0, 1.0, 1.0]).view(1, 1, 4)))
        self.assertTrue(torch.allclose(yb, torch.full((1, 2, 4), 0.5)))

    def test_reweight_allows_amplification(self):
        gate = VisualTokenGate(("a",), 2, mode="reweight")
        with torch.no_grad():
            gate.gamma[0] = torch.tensor([2.0, 0.5])
        y = gate(torch.ones(1, 1, 2), "a")
        self.assertTrue(torch.allclose(y, torch.tensor([2.0, 0.5]).view(1, 1, 2)))


if __name__ == "__main__":
    unittest.main()
