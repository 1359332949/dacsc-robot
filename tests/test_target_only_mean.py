import unittest

import numpy as np
import torch

from dacsc.target_only_mean import (
    ConditionalSourceBaseline,
    SourceResidualProjector,
    TargetOnlyMeanEstimator,
    fit_source_calibration,
)


class TestConditionalSourceBaseline(unittest.TestCase):
    def test_ridge_recovers_linear_source(self):
        rng = np.random.RandomState(0)
        x = rng.randn(200, 4)
        w = rng.randn(4, 3)
        y = x @ w + 2.0
        base = ConditionalSourceBaseline(l2=1e-6)
        base.fit(x, y)
        pred = base.predict(x)
        self.assertLess(float(np.abs(pred - y).mean()), 1e-3)

    def test_standardize_switch_is_effective(self):
        rng = np.random.RandomState(1)
        x = rng.randn(40, 3) * np.array([0.01, 10.0, 100.0])
        y = x @ rng.randn(3, 2)
        standardized = ConditionalSourceBaseline(l2=1.0, standardize=True)
        raw = ConditionalSourceBaseline(l2=1.0, standardize=False)
        standardized.fit(x, y)
        raw.fit(x, y)
        self.assertFalse(np.allclose(standardized.scale, raw.scale))

    def test_rejects_non_finite_fit_data(self):
        x = np.ones((3, 2))
        y = np.ones((3, 4))
        y[0, 0] = np.nan
        with self.assertRaises(ValueError):
            ConditionalSourceBaseline().fit(x, y)


class TestSourceResidualProjector(unittest.TestCase):
    def test_removes_source_mean_and_content_direction(self):
        source_residuals = np.array([
            [1.0, -1.0, 0.0],
            [3.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 2.0, 0.0],
        ])
        projector = SourceResidualProjector.fit(source_residuals, rank=1)
        target_residual = torch.tensor([[4.0, 0.0, 5.0]], dtype=torch.float64)
        filtered = projector.transform(target_residual)
        basis = torch.as_tensor(projector.basis, dtype=torch.float64)
        self.assertLess(float((filtered @ basis).abs().max()), 1e-10)
        self.assertGreater(float(filtered[0, 2]), 4.9)

    def test_rank_zero_only_subtracts_source_residual_mean(self):
        residuals = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]])
        projector = SourceResidualProjector.fit(residuals, rank=0)
        x = torch.tensor([[5.0, 7.0]])
        y = projector.transform(x)
        # Source residual mean is [2, 3], so [5, 7] -> [3, 4].
        self.assertTrue(torch.allclose(y, torch.tensor([[3.0, 4.0]])))


class TestTargetOnlyMeanEstimator(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        x = rng.randn(80, 4)
        w = rng.randn(4, 3)
        self.base = ConditionalSourceBaseline(l2=1e-6)
        self.base.fit(x, x @ w)
        self.context = np.array([[0.1, 0.2, 0.3, 0.4]])

    def test_target_only_residual(self):
        est = TargetOnlyMeanEstimator(self.base, momentum=0.5, alpha=1.0)
        source = self.base.predict(self.context)
        visual = torch.tensor(source + np.array([[1.0, 2.0, 3.0]]), dtype=torch.float32)
        res = est.residual(visual, self.context)
        self.assertTrue(torch.allclose(res, torch.tensor([[1.0, 2.0, 3.0]]), atol=1e-4))

    def test_episode_reset_clears_history(self):
        est = TargetOnlyMeanEstimator(self.base, momentum=0.5)
        source = torch.tensor(self.base.predict(self.context), dtype=torch.float32)
        first = torch.tensor([[1.0, 2.0, 3.0]])
        est.update(source + first, self.context)
        est.reset()
        second = torch.tensor([[4.0, 5.0, 6.0]])
        out = est.update(source + second, self.context)
        self.assertTrue(torch.allclose(out, second))

    def test_content_projector_is_applied_before_ema(self):
        projector = SourceResidualProjector.fit(
            np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            rank=1,
        )
        est = TargetOnlyMeanEstimator(self.base, residual_projector=projector)
        raw = torch.tensor([[1.0, 2.0, 3.0]])
        source = torch.tensor(self.base.predict(self.context), dtype=torch.float32)
        out = est.update(source + raw, self.context)
        basis = torch.as_tensor(projector.basis, dtype=out.dtype)
        self.assertLess(float((out @ basis).abs().max()), 1e-5)

    def test_clip(self):
        est = TargetOnlyMeanEstimator(self.base, alpha=1.0, clip_norm=0.5)
        source = torch.tensor(self.base.predict(self.context), dtype=torch.float32)
        vec = torch.tensor([[10.0, 10.0, 10.0]])
        corr = est.correction(source + vec, self.context)
        self.assertLessEqual(float(corr.norm()), 0.5001)

    def test_rejects_batch_size_change_without_reset(self):
        est = TargetOnlyMeanEstimator(self.base, momentum=0.5)
        source = torch.tensor(self.base.predict(self.context), dtype=torch.float32)
        est.update(source, self.context)
        with self.assertRaises(ValueError):
            est.update(source.repeat(2, 1), np.repeat(self.context, 2, axis=0))


class TestSourceCalibration(unittest.TestCase):
    def test_cross_fitted_calibration(self):
        rng = np.random.RandomState(42)
        x = rng.randn(20, 5)
        w = rng.randn(5, 7)
        y = x @ w + rng.randn(20, 7) * 0.05
        base, projector, diagnostics = fit_source_calibration(
            x, y, l2=1.0, content_rank=2, folds=5, seed=3
        )
        self.assertEqual(base.predict(x[:2]).shape, (2, 7))
        self.assertEqual(projector.basis.shape, (7, 2))
        self.assertEqual(diagnostics["source_rows"], 20)


if __name__ == "__main__":
    unittest.main()
