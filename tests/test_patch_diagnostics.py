import unittest
import numpy as np

from dacsc.patch_diagnostics import (
    cosine_self_similarity,
    normalized_distance,
    random_clean_donor,
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
