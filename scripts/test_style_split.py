#!/usr/bin/env python3
"""Fast unit checks for the style allowlist and split cache signature."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dacsc.sampler import DomainBalancedSampler


class _Sub:
    dataset_path = "/tmp/fake/lerobot"
    domain_map = {}

    def __init__(self):
        # Four steps per style plus one unlabeled step.
        self.styles = [11, 11, 30, 30, 31, 31, 40, 40, 41, 41, 50, 50, None]
        self.domain_map = {i: (s, 11) for i, s in enumerate(self.styles)}


class _FakeDataset:
    def __init__(self):
        self.sub = _Sub()
        self.datasets = [self.sub]

    def __len__(self):
        return len(self.sub.styles)

    def sample_step(self, i):
        return self.sub, i, 0


def main():
    ds = _FakeDataset()
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DACSC_CACHE_DIR"] = tmp
        train = DomainBalancedSampler(
            ds, batch_size=4, domain_id_min=11, domain_group_size=20,
            samples_per_domain=1, max_groups_per_batch=1, seed=42,
            allowed_styles=set(range(11, 31)))
        train._build()
        emitted = set(train._all)
        expected = {i for i, style in enumerate(ds.sub.styles)
                    if style in set(range(11, 31))}
        assert emitted == expected, (emitted, expected)
        assert set(train._groups) == {0}, train._groups
        yielded = list(train)
        assert yielded and set(yielded) <= expected
        assert len(train) == 4

        val = DomainBalancedSampler(
            ds, batch_size=4, domain_id_min=11, domain_group_size=20,
            samples_per_domain=1, max_groups_per_batch=1, seed=42,
            allowed_styles=set(range(31, 41)))
        assert val._cache_path() != train._cache_path()
        val._build()
        assert set(val._groups) == {1}, val._groups
        assert set(val._all) == {4, 5, 6, 7}

    print("STYLE_SPLIT_SAMPLER_OK")


if __name__ == "__main__":
    main()
