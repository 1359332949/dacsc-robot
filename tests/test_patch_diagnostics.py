import unittest
import numpy as np

from scripts.analyze_patch_pairs import analyze_view_window
from dacsc.patch_diagnostics import (
    cosine_self_similarity,
    normalized_distance,
    random_clean_donor,
    resolve_token_layout,
    state_balanced_mean,
    token_descriptor,
    validate_patch_dict,
)


class TestPatchDiagnostics(unittest.TestCase):
    def test_patch_shapes_are_runtime_driven(self):
        # Both 196 and 256 tokens are accepted; no fixed 256-token assumption.
        for n_tokens in (196, 256, 324):
            patches = {
                "agentview": np.zeros((1, n_tokens, 384), dtype=np.float32),
                "eye_in_hand": np.ones((1, n_tokens, 384), dtype=np.float32),
            }
            shapes = validate_patch_dict(patches, ("agentview", "eye_in_hand"))
            self.assertEqual(shapes["agentview"], [1, n_tokens, 384])

    def test_patch_validation_rejects_bad_rank_and_nonfinite(self):
        with self.assertRaises(ValueError):
            validate_patch_dict({"agentview": np.zeros((256, 384), dtype=np.float32)})
        bad = np.zeros((1, 4, 8), dtype=np.float32)
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_patch_dict({"agentview": bad})

    def test_descriptor_and_self_similarity_keep_dynamic_token_count(self):
        tokens = np.arange(7 * 5, dtype=np.float32).reshape(1, 7, 5)
        self.assertEqual(token_descriptor(tokens).shape, (10,))
        sim = cosine_self_similarity(tokens)
        self.assertEqual(sim.shape, (7, 7))
        np.testing.assert_allclose(np.diag(sim), np.ones(7), atol=1e-6)

    def test_normalized_distance(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 4.0, 3.0])
        self.assertAlmostEqual(normalized_distance(a, b), np.sqrt(4.0 / 3.0))
        self.assertAlmostEqual(normalized_distance(a, b, np.ones(3)), np.sqrt(4.0 / 3.0))

    def test_phase1_analyzer_uses_disjoint_state_roles_and_returns_metrics(self):
        rng = np.random.default_rng(17)
        records = []
        clean, shift = [], []
        row = 0
        for role, state_ids in (
            ("source_fit", range(4)),
            ("source_validation", range(10, 14)),
        ):
            for state_id in state_ids:
                base = rng.normal(size=(5, 8)).astype(np.float32)
                shifted = base + np.linspace(0.15, 0.45, 8, dtype=np.float32)[None, :]
                clean.append(base)
                shift.append(shifted)
                records.append({
                    "feature_row": row,
                    "state_id": state_id,
                    "role": role,
                    "window": "early",
                    "chunk_index": 0,
                    "pairing_pass": True,
                })
                row += 1
        arrays = {
            "clean_agentview": np.stack(clean),
            "shift_agentview": np.stack(shift),
        }
        report = analyze_view_window(
            arrays, records, view="agentview", window="early", seed=3
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["source_fit_states"], 4)
        self.assertEqual(report["source_validation_states"], 4)
        self.assertFalse(
            report["heldout_state_domain_separation"]["source_fit_validation_state_overlap"]
        )
        self.assertGreater(report["R_sep"], 0.0)
        self.assertGreaterEqual(
            report["heldout_state_domain_separation"]["logistic_regression_AUROC"],
            0.5,
        )

    def test_token_layout_resolver_patch_only_and_prefix(self):
        grid = [{
            "module": "patch_embed.proj",
            "expected_grid_hw": [16, 16],
            "expected_grid_tokens": 256,
            "actual_tokens_per_view": 256,
        }]
        patch_only = resolve_token_layout(grid, [])
        self.assertTrue(patch_only["patch_layout_verified"])
        self.assertEqual(patch_only["token_layout"], "patch_tokens_only")
        self.assertEqual(patch_only["patch_token_count_per_view"], 256)
        self.assertEqual(patch_only["prefix_token_count"], 0)

        with_prefix = [{
            **grid[0],
            "actual_tokens_per_view": 261,
        }]
        prefix = resolve_token_layout(with_prefix, [{"num_prefix_tokens": 5}])
        self.assertTrue(prefix["patch_layout_verified"])
        self.assertEqual(prefix["token_layout"], "patch_tokens_plus_prefix")
        self.assertEqual(prefix["prefix_token_count"], 5)

    def test_token_layout_resolver_fails_on_unexplained_or_ambiguous_count(self):
        unexplained = resolve_token_layout([{
            "module": "patch_embed.proj",
            "expected_grid_hw": [16, 16],
            "expected_grid_tokens": 256,
            "actual_tokens_per_view": 257,
        }], [])
        self.assertFalse(unexplained["patch_layout_verified"])

        ambiguous = resolve_token_layout([
            {
                "module": "patch_embed_a",
                "expected_grid_hw": [16, 16],
                "expected_grid_tokens": 256,
                "actual_tokens_per_view": 256,
            },
            {
                "module": "patch_embed_b",
                "expected_grid_hw": [8, 32],
                "expected_grid_tokens": 256,
                "actual_tokens_per_view": 256,
            },
        ], [])
        self.assertFalse(ambiguous["patch_layout_verified"])

    def test_token_layout_resolver_infers_cls_and_register_prefix(self):
        layout = resolve_token_layout([{
            "module": "patch_embed.proj",
            "expected_grid_hw": [16, 16],
            "expected_grid_tokens": 256,
            "actual_tokens_per_view": 261,
        }], [{
            "cls_token_shape": [1, 1, 384],
            "register_tokens_shape": [1, 4, 384],
        }])
        self.assertTrue(layout["patch_layout_verified"])
        self.assertEqual(layout["prefix_token_count"], 5)

    def test_state_balanced_mean_weights_states_not_chunk_counts(self):
        records = [
            {"state_id": 1, "metric": {"value": 1.0}},
            {"state_id": 1, "metric": {"value": 3.0}},
            {"state_id": 1, "metric": {"value": 5.0}},
            {"state_id": 2, "metric": {"value": 9.0}},
        ]
        out = state_balanced_mean(records, "metric")
        self.assertEqual(out["n_states"], 2)
        self.assertEqual(out["n_records"], 4)
        self.assertAlmostEqual(out["state_balanced_mean"]["value"], 6.0)
        self.assertAlmostEqual(out["per_state"][1]["value"], 3.0)
        self.assertAlmostEqual(out["per_state"][2]["value"], 9.0)

    def test_state_balanced_mean_supports_nested_metrics(self):
        records = [
            {"state_id": 1, "metrics": {"distance": {"value": 2.0}}},
            {"state_id": 2, "metrics": {"distance": {"value": 4.0}}},
        ]
        out = state_balanced_mean(records, "metrics", "distance")
        self.assertAlmostEqual(out["state_balanced_mean"]["value"], 3.0)

    def test_random_donor_is_different_state_and_matched_window(self):
        records = [
            {"state_id": 1, "role": "source_fit", "window": "early", "chunk_index": 0, "pairing_pass": True},
            {"state_id": 2, "role": "source_fit", "window": "early", "chunk_index": 0, "pairing_pass": True},
            {"state_id": 3, "role": "source_fit", "window": "late", "chunk_index": 6, "pairing_pass": True},
            {"state_id": 4, "role": "source_validation", "window": "early", "chunk_index": 0, "pairing_pass": True},
        ]
        donor = random_clean_donor(records, 0, seed=7)
        self.assertEqual(donor, 1)
        self.assertIsNone(random_clean_donor(records, 2, seed=7))


if __name__ == "__main__":
    unittest.main()
