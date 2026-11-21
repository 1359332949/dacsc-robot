import unittest
import numpy as np
from scripts.b3_oracle10 import fold_predictions, ridge, predict

class FoldIsolation(unittest.TestCase):
    def test_test_labels_cannot_change_predictions_constant_or_beta(self):
        rng = np.random.default_rng(42)
        x, y = rng.normal(size=(24, 5)), rng.normal(size=(24, 7))
        states = np.repeat([1, 2, 3, 4], 6)
        before = fold_predictions(x, y, states, 4)
        poisoned = y.copy(); poisoned[states == 4] += 10000
        after = fold_predictions(x, poisoned, states, 4)
        for a, b in zip(before[:3], after[:3]): np.testing.assert_array_equal(a, b)
        self.assertEqual(before[3], after[3])
        np.testing.assert_allclose(before[1][0], y[states != 4].mean(0))
        self.assertNotIn(4, before[3]['train_states'])

    def test_matches_legacy_unpenalized_intercept(self):
        rng = np.random.default_rng(7)
        for dim in (4, 30):
            x, y = rng.normal(size=(20, dim)), rng.normal(size=(20, 7))
            mu, sd = x.mean(0), x.std(0)+1e-8
            d = np.c_[(x-mu)/sd, np.ones(len(x))]
            penalty = 100*np.eye(dim+1); penalty[-1,-1] = 0
            legacy = d @ np.linalg.solve(d.T@d+penalty, d.T@y)
            np.testing.assert_allclose(predict(x, ridge(x, y)), legacy, atol=1e-12)

if __name__ == '__main__': unittest.main()
