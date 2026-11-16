import unittest
import numpy as np
import torch
from dacsc.target_only_mean import ConditionalSourceBaseline, TargetOnlyMeanEstimator


class TestTargetOnlyMean(unittest.TestCase):
    def test_ridge_recovers_linear_source(self):
        rng = np.random.RandomState(0)
        x = rng.randn(200, 4)
        w = rng.randn(4, 3)
        y = x @ w + 2.0
        base = ConditionalSourceBaseline(l2=1e-6)
        base.fit(x, y)
        pred = base.predict(x)
        self.assertLess(float(np.abs(pred - y).mean()), 1e-3)

    def test_target_only_residual(self):
        rng = np.random.RandomState(0)
        x = rng.randn(50, 4); w = rng.randn(4, 3)
        base = ConditionalSourceBaseline(l2=1e-6); base.fit(x, x @ w)
        est = TargetOnlyMeanEstimator(base, momentum=0.5, alpha=1.0)
        ctx = np.array([[0.1, 0.2, 0.3, 0.4]])
        source = base.predict(ctx)
        visual = torch.tensor(source + np.array([[1.0, 2.0, 3.0]]), dtype=torch.float32)
        res = est.residual(visual, ctx)
        self.assertTrue(torch.allclose(res, torch.tensor([[1.0, 2.0, 3.0]]), atol=1e-4))
        corr = est.correction(visual, ctx)
        self.assertTrue(torch.allclose(corr, res, atol=1e-4))

    def test_clip(self):
        rng = np.random.RandomState(1)
        x = rng.randn(20, 2); base = ConditionalSourceBaseline(l2=1e-6); base.fit(x, x[:, :3] if False else x @ np.eye(2))
        est = TargetOnlyMeanEstimator(base, alpha=1.0, clip_norm=0.5)
        ctx = np.array([[0.0, 0.0]])
        vec = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
        corr = est.correction(vec, ctx)
        self.assertLessEqual(float(corr.norm()), 0.5001)


if __name__ == "__main__":
    unittest.main()
