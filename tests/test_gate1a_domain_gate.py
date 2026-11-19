import unittest
import numpy as np
import numpy as np
from scripts.gate1a_domain_gate import score, state_desc


class TestGate1ADomainScore(unittest.TestCase):
    def test_mean_and_logstd_are_separate_components(self):
        prior_mu = np.zeros(4)
        prior_mu_scale = np.ones(4)
        prior_logsd = np.zeros(4)
        mean_shift = score((np.ones(4), np.zeros(4)), prior_mu, prior_mu_scale, prior_logsd)
        std_shift = score((np.zeros(4), np.ones(4)), prior_mu, prior_mu_scale, prior_logsd)
        self.assertAlmostEqual(mean_shift, 1.0, places=5)
        self.assertAlmostEqual(std_shift, 1.0, places=5)

    def test_failed_pair_is_rejected(self):
        npz = {'clean_v': np.zeros((1, 1, 2))}
        records = [{'role': 'source_fit', 'window': 'early', 'state_id': 0,
                    'chunk_index': 0, 'feature_row': 0, 'pairing_pass': False}]
        with self.assertRaises(RuntimeError):
            state_desc(npz, records, 'v', 'clean', 'early')

    def test_zero_for_exact_prior(self):
        prior_mu = np.array([0.2, -0.1, 0.4])
        scale = np.array([0.5, 0.7, 0.9])
        logsd = np.array([-1.0, -2.0, -0.5])
        self.assertAlmostEqual(score((prior_mu, logsd), prior_mu, scale, logsd), 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
