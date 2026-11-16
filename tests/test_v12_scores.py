import importlib.util
import unittest
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[1] / "scripts" / "v12_scores.py"
spec = importlib.util.spec_from_file_location("v12_scores", path)
v12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v12)


class TestV12Scores(unittest.TestCase):
    def test_pair_score_per_dim_and_no_view_mix(self):
        clean_a = np.zeros((2, 3, 4)); shift_a = np.ones((2, 3, 4))
        clean_b = np.zeros((2, 3, 4)); shift_b = np.full((2, 3, 4), 10.0)
        a = v12.pair_score(clean_a, shift_a)
        b = v12.pair_score(clean_b, shift_b)
        self.assertEqual(a.shape, (4,))
        self.assertTrue(np.allclose(a, 1.0))
        self.assertTrue(np.allclose(b, 10.0))
        # no cross-view averaging: separate calls stay separate
        self.assertFalse(np.allclose(a, b))

    def test_stat_primary_and_alt_dims(self):
        x = np.random.RandomState(0).randn(5, 7, 4)
        p = v12.stat_primary(x); a = v12.stat_alt_score(x)
        self.assertEqual(p.shape, (4,)); self.assertEqual(a.shape, (4,))
        # primary = std over states of spatial mean
        self.assertTrue(np.allclose(p, x.mean(axis=1).std(axis=0)))
        # alt = spatial mean of per-patch state std
        self.assertTrue(np.allclose(a, x.std(axis=0).mean(axis=0)))

    def test_valid_positions_filters_failed_and_maps(self):
        manifest = {"records": [
            {"index": 0, "compare_pass": True}, {"index": 1, "compare_pass": False},
            {"index": 2, "compare_pass": True}]}
        valid, pos = v12.valid_positions(manifest)
        self.assertEqual(valid, [0, 2])
        self.assertEqual(pos, {0: 0, 2: 1})

    def test_pair_score_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            v12.pair_score(np.zeros((2, 3, 4)), np.zeros((2, 2, 4)))


if __name__ == "__main__":
    unittest.main()
