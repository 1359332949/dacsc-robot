import unittest
from types import SimpleNamespace

import torch

from scripts.probe_style_linear import (
    fit_linear_probe,
    parse_ints,
    sample_source_frames,
    stable_bucket,
)


class StyleProbeTest(unittest.TestCase):
    def test_parse_integer_ranges_and_holdout_buckets(self):
        self.assertEqual(parse_ints(["2", "5-7"]), [2, 5, 6, 7])
        self.assertEqual(stable_bucket("same"), stable_bucket("same"))
        self.assertIn(stable_bucket("same"), range(5))

    def test_sampling_selects_frames_without_duplicating_an_episode(self):
        flat = []
        for step in range(10):
            flat.append((0, step, 11, "instruction", "TaskA", 3, step, 10))
        dataset = SimpleNamespace(flat=flat)
        rows = sample_source_frames(dataset, frames_per_episode=2, seed=42)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({row["step"] for row in rows}), 2)
        self.assertEqual({row["style"] for row in rows}, {11})

    def test_linear_probe_learns_separable_frozen_features(self):
        torch.manual_seed(1)
        centers = torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
            ]
        )
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        features = centers[labels] + 0.05 * torch.randn(9, 4)
        metrics = fit_linear_probe(
            features,
            labels,
            torch.tensor([True] * 6 + [False] * 3),
            torch.tensor([False] * 6 + [True] * 3),
            num_classes=3,
            steps=200,
            seed=3,
        )
        self.assertGreater(metrics["balanced_accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
