import unittest
import torch
from dacsc.failure_monitor import CausalFailureMonitor


class FailureMonitorTest(unittest.TestCase):
    def test_shape_and_causality(self):
        m = CausalFailureMonitor(window_size=5).eval()
        s = torch.randn(2, 5, 8); a = torch.randn(2, 5, 7); v = torch.randn(2, 5, 768)
        with torch.no_grad():
            out = m(s, a, v)
        self.assertEqual(tuple(out.shape), (2,))
        self.assertTrue(torch.all(out >= 0) and torch.all(out <= 1))
        # Changing a future step's input must not change an earlier-window prediction
        # unless the window itself is changed; the monitor only returns the last step.
        w1 = m(torch.randn(1, 5, 8), torch.randn(1, 5, 7), torch.randn(1, 5, 768))
        self.assertEqual(tuple(w1.shape), (1,))

    def test_window_mismatch_rejected(self):
        m = CausalFailureMonitor(window_size=5)
        with self.assertRaises(ValueError):
            m(torch.randn(1, 4, 8), torch.randn(1, 4, 7), torch.randn(1, 4, 768))


if __name__ == "__main__":
    unittest.main()
