import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dacsc.action_guard import limit_rotation_delta
from dacsc.target_only_mean import (
    ConditionalSourceBaseline,
    SourceResidualConfidenceGate,
    SourceResidualProjector,
    TargetOnlyMeanEstimator,
    fit_source_calibration,
)


class IdentityNormalizer:
    def unnormalize_action(self, action):
        return action

    def normalize_action(self, action):
        return action


class CorpLikeNormalizer:
    """Matches the checkpoint Normalizer API (no normalize_action method)."""

    def __init__(self):
        self.action_min = torch.tensor([-2., -2., -2., -1., -1., -1., -1.])
        self.action_max = torch.tensor([2., 2., 2., 1., 1., 1., 1.])

    def unnormalize_action(self, action):
        low = self.action_min.to(action.device, action.dtype)
        high = self.action_max.to(action.device, action.dtype)
        out = action.clone()
        out[..., :6] = 0.5 * (out[..., :6] + 1.0) * (high[:6] - low[:6]) + low[:6]
        return out


class TestConditionalSourceBaseline(unittest.TestCase):
    def test_ridge_recovers_linear_source(self):
        rng = np.random.RandomState(0)
        x = rng.randn(200, 4)
        w = rng.randn(4, 3)
        y = x @ w + 2.0
        base = ConditionalSourceBaseline(l2=1e-6)
        base.fit(x, y)
        self.assertLess(float(np.abs(base.predict(x) - y).mean()), 1e-3)

    def test_standardize_switch_is_effective(self):
        rng = np.random.RandomState(1)
        x = rng.randn(40, 3) * np.array([0.01, 10.0, 100.0])
        y = x @ rng.randn(3, 2)
        a = ConditionalSourceBaseline(l2=1.0, standardize=True)
        b = ConditionalSourceBaseline(l2=1.0, standardize=False)
        a.fit(x, y); b.fit(x, y)
        self.assertFalse(np.allclose(a.scale, b.scale))

    def test_rejects_non_finite_fit_data(self):
        x = np.ones((3, 2)); y = np.ones((3, 4)); y[0, 0] = np.nan
        with self.assertRaises(ValueError):
            ConditionalSourceBaseline().fit(x, y)


class TestSourceResidualProjector(unittest.TestCase):
    def test_removes_source_mean_and_content_direction(self):
        source_residuals = np.array([
            [1.0, -1.0, 0.0], [3.0, 1.0, 0.0],
            [2.0, 0.0, 0.0], [4.0, 2.0, 0.0],
        ])
        projector = SourceResidualProjector.fit(source_residuals, rank=1)
        target = torch.tensor([[4.0, 0.0, 5.0]], dtype=torch.float64)
        filtered = projector.transform(target)
        basis = torch.as_tensor(projector.basis, dtype=torch.float64)
        self.assertLess(float((filtered @ basis).abs().max()), 1e-10)
        self.assertGreater(float(filtered[0, 2]), 4.9)

    def test_rank_zero_only_subtracts_source_residual_mean(self):
        residuals = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]])
        projector = SourceResidualProjector.fit(residuals, rank=0)
        actual = projector.transform(torch.tensor([[5.0, 7.0]]))
        self.assertTrue(torch.allclose(actual, torch.tensor([[3.0, 4.0]])))


class TestSourceResidualConfidenceGate(unittest.TestCase):
    def setUp(self):
        self.contexts = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
            [0.2, 0.0], [0.0, 0.2], [0.2, 0.2], [0.15, 0.05],
        ])
        self.residuals = np.array([
            [0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1],
            [0.05, 0.05], [-0.05, -0.05], [0.08, -0.02], [-0.08, 0.02],
        ])
        self.gate = SourceResidualConfidenceGate.fit(self.contexts, self.residuals)

    def test_diagnostics_separate_residual_and_context_rejection(self):
        inside = torch.tensor([[0.05, 0.05]])
        large = torch.tensor([[2.0, 1.0]])
        in_diag = self.gate.diagnostics(large, inside)
        self.assertGreater(float(in_diag["residual_confidence"]), 0.0)
        self.assertGreater(float(in_diag["context_confidence"]), 0.0)

        far = torch.tensor([[20.0, -20.0]])
        far_diag = self.gate.diagnostics(large, far)
        self.assertGreater(float(far_diag["residual_confidence"]), 0.0)
        self.assertEqual(float(far_diag["context_confidence"]), 0.0)
        self.assertIn("context_outside_source_support", far_diag["abstain_reasons"][0])

        small = torch.tensor([[0.02, -0.01]])
        small_diag = self.gate.diagnostics(small, inside)
        self.assertEqual(float(small_diag["residual_confidence"]), 0.0)
        self.assertGreater(float(small_diag["context_confidence"]), 0.0)
        self.assertIn("residual_within_source_noise", small_diag["abstain_reasons"][0])

    def test_rejected_ood_observation_holds_last_accepted_correction(self):
        contexts = np.array([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
            [0.2, 0.0], [0.0, 0.2], [0.2, 0.2], [0.15, 0.05],
        ])
        residuals = np.array([
            [0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1],
            [0.05, 0.05], [-0.05, -0.05], [0.08, -0.02], [-0.08, 0.02],
        ])
        source = ConditionalSourceBaseline(l2=1.0)
        source.fit(contexts, np.zeros((8, 2)))
        gate = SourceResidualConfidenceGate.fit(contexts, residuals)
        est = TargetOnlyMeanEstimator(
            source, confidence_gate=gate, min_update_confidence=0.25
        )

        near = contexts[[0]]
        est.update(torch.tensor([[20.0, 0.0]]), near)
        self.assertGreater(float(est.last_confidence), 0.25)
        accepted_correction = est.applied_correction().clone()
        self.assertGreater(float(accepted_correction.norm()), 0.0)

        far = np.array([[100.0, -100.0]])
        est.update(torch.tensor([[40.0, 0.0]]), far)
        self.assertEqual(float(est.last_confidence), 0.0)
        self.assertFalse(bool(est.last_update_accepted))
        self.assertTrue(torch.allclose(est.applied_correction(), accepted_correction))
        self.assertIn(
            "context_outside_source_support",
            est.last_gate_diagnostics["abstain_reasons"][0],
        )

    def test_rejected_clean_like_observation_preserves_identity(self):
        source = ConditionalSourceBaseline(l2=1.0)
        source.fit(self.contexts, np.zeros((len(self.contexts), 2)))
        est = TargetOnlyMeanEstimator(
            source, confidence_gate=self.gate, min_update_confidence=0.25
        )
        context = self.contexts[[0]]
        prediction = torch.tensor(source.predict(context), dtype=torch.float32)
        est.update(prediction + torch.tensor([[0.02, -0.01]]), context)
        self.assertFalse(bool(est.last_update_accepted))
        self.assertTrue(torch.equal(est.applied_correction(), torch.zeros(1, 2)))

    def test_source_noise_residual_is_abstained(self):
        confidence = self.gate.confidence(
            torch.tensor([[0.02, -0.01]]), torch.tensor([[0.05, 0.05]])
        )
        self.assertEqual(float(confidence), 0.0)

    def test_large_residual_inside_support_is_admitted(self):
        confidence = self.gate.confidence(
            torch.tensor([[2.0, 1.0]]), torch.tensor([[0.05, 0.05]])
        )
        self.assertGreater(float(confidence), 0.0)

    def test_out_of_support_context_abstains(self):
        confidence = self.gate.confidence(
            torch.tensor([[2.0, 1.0]]), torch.tensor([[20.0, -20.0]])
        )
        self.assertEqual(float(confidence), 0.0)


class TestTargetOnlyMeanEstimator(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        x = rng.randn(80, 4); w = rng.randn(4, 3)
        self.base = ConditionalSourceBaseline(l2=1e-6)
        self.base.fit(x, x @ w)
        self.context = np.array([[0.1, 0.2, 0.3, 0.4]])

    def test_target_only_residual(self):
        source = torch.tensor(self.base.predict(self.context), dtype=torch.float32)
        est = TargetOnlyMeanEstimator(self.base)
        residual = est.residual(source + torch.tensor([[1.0, 2.0, 3.0]]), self.context)
        self.assertTrue(torch.allclose(residual, torch.tensor([[1.0, 2.0, 3.0]]), atol=1e-4))

    def test_episode_reset_clears_history(self):
        source = torch.tensor(self.base.predict(self.context), dtype=torch.float32)
        est = TargetOnlyMeanEstimator(self.base, momentum=0.5)
        est.update(source + torch.tensor([[1.0, 2.0, 3.0]]), self.context)
        est.reset()
        out = est.update(source + torch.tensor([[4.0, 5.0, 6.0]]), self.context)
        self.assertTrue(torch.allclose(out, torch.tensor([[4.0, 5.0, 6.0]])))

    def test_content_projector_is_applied_before_ema(self):
        projector = SourceResidualProjector.fit(
            np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]]), rank=1
        )
        source = torch.tensor(self.base.predict(self.context), dtype=torch.float32)
        est = TargetOnlyMeanEstimator(self.base, residual_projector=projector)
        out = est.update(source + torch.tensor([[1., 2., 3.]]), self.context)
        basis = torch.as_tensor(projector.basis, dtype=out.dtype)
        self.assertLess(float((out @ basis).abs().max()), 1e-5)

    def test_confidence_gate_makes_clean_like_residual_identity(self):
        contexts = np.array([
            [0.0, 0.1, 0.2, 0.3], [0.1, 0.1, 0.2, 0.3],
            [0.0, 0.2, 0.2, 0.3], [0.1, 0.2, 0.2, 0.3],
            [0.2, 0.1, 0.3, 0.4], [0.0, 0.3, 0.3, 0.4],
            [0.2, 0.3, 0.3, 0.4], [0.15, 0.25, 0.3, 0.4],
        ])
        residuals = np.array([
            [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0], [0.0, -0.1, 0.0],
            [0.05, 0.05, 0.0], [-0.05, -0.05, 0.0],
            [0.08, -0.02, 0.0], [-0.08, 0.02, 0.0],
        ])
        gate = SourceResidualConfidenceGate.fit(contexts, residuals)
        source = torch.tensor(self.base.predict(self.context), dtype=torch.float32)
        est = TargetOnlyMeanEstimator(self.base, confidence_gate=gate)
        est.update(source + torch.tensor([[0.02, -0.01, 0.0]]), self.context)
        self.assertTrue(torch.equal(est.applied_correction(), torch.zeros(1, 3)))

    def test_rejects_batch_size_change_without_reset(self):
        source = torch.tensor(self.base.predict(self.context), dtype=torch.float32)
        est = TargetOnlyMeanEstimator(self.base)
        est.update(source, self.context)
        with self.assertRaises(ValueError):
            est.update(source.repeat(2, 1), np.repeat(self.context, 2, axis=0))


class TestSourceCalibration(unittest.TestCase):
    def test_cross_fitted_calibration(self):
        rng = np.random.RandomState(42)
        x = rng.randn(20, 5); w = rng.randn(5, 7)
        y = x @ w + rng.randn(20, 7) * 0.05
        base, projector, gate, diagnostics = fit_source_calibration(
            x, y, l2=1.0, content_rank=2, folds=5, seed=3
        )
        self.assertEqual(base.predict(x[:2]).shape, (2, 7))
        self.assertEqual(projector.basis.shape, (7, 2))
        self.assertEqual(gate.source_contexts_z.shape, x.shape)
        self.assertEqual(diagnostics["source_rows"], 20)


class TestActionRotationGuard(unittest.TestCase):
    def test_rotation_budget_is_respected_and_other_dims_unchanged(self):
        reference = torch.zeros(1, 2, 7)
        candidate = torch.zeros(1, 2, 7)
        candidate[..., :3] = 0.3
        candidate[..., 3:6] = torch.tensor([0.0, 0.0, 0.4])
        candidate[..., 6] = 1.0
        bounded = limit_rotation_delta(candidate, reference, IdentityNormalizer(), 0.05)
        self.assertTrue(torch.allclose(bounded[..., :3], candidate[..., :3]))
        self.assertTrue(torch.allclose(bounded[..., 6], candidate[..., 6]))
        r = Rotation.from_rotvec(bounded[..., 3:6].numpy().reshape(-1, 3))
        a = r.magnitude()
        self.assertLessEqual(float(a.max()), 0.050001)

    def test_zero_rotation_budget_preserves_reference_rotation(self):
        reference = torch.zeros(1, 1, 7)
        candidate = torch.zeros(1, 1, 7)
        candidate[..., 3:6] = torch.tensor([0.2, -0.1, 0.3])
        bounded = limit_rotation_delta(candidate, reference, IdentityNormalizer(), 0.0)
        self.assertTrue(torch.equal(bounded[..., 3:6], reference[..., 3:6]))

    def test_supports_corp_checkpoint_normalizer_api(self):
        normalizer = CorpLikeNormalizer()
        reference = torch.zeros(1, 1, 7)
        candidate = torch.zeros(1, 1, 7)
        candidate[..., :3] = torch.tensor([0.2, -0.1, 0.3])
        candidate[..., 3:6] = torch.tensor([0.0, 0.0, 0.5])
        candidate[..., 6] = 1.0
        bounded = limit_rotation_delta(candidate, reference, normalizer, 0.05)
        restored = normalizer.unnormalize_action(bounded)
        relative = Rotation.from_rotvec(restored[..., 3:6].numpy().reshape(-1, 3))
        self.assertLessEqual(float(relative.magnitude().max()), 0.050001)
        self.assertTrue(torch.allclose(bounded[..., :3], candidate[..., :3], atol=1e-6))
        self.assertTrue(torch.equal(bounded[..., 6], candidate[..., 6]))


if __name__ == "__main__":
    unittest.main()
