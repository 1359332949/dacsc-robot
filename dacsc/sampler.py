"""按粗粒度外观域平衡采样的 sampler。"""
from __future__ import annotations
import hashlib
import os
import pickle
import random
from collections import defaultdict
from torch.utils.data import Sampler


class DomainBalancedSampler(Sampler):
    """每个 batch 尽量覆盖所有粗域，每域至少 2–3 个样本。

    通过 dataset.sample_step(index) 离线建立 index -> style_group 映射，
    然后把 sampler 的索引顺序排成 domain-balanced 的 batch。
    """
    def __init__(self, dataset, batch_size: int = 32, domain_id_min: int = 11,
                 domain_group_size: int = 5, samples_per_domain: int = 3,
                 max_groups_per_batch: int | None = None, seed: int = 42,
                 allowed_styles: set[int] | None = None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.domain_id_min = int(domain_id_min)
        self.domain_group_size = max(int(domain_group_size), 1)
        self.samples_per_domain = int(samples_per_domain)
        self.max_groups_per_batch = (None if max_groups_per_batch is None
                                     else int(max_groups_per_batch))
        self.seed = int(seed)
        self.allowed_styles = (None if allowed_styles is None
                               else {int(s) for s in allowed_styles})
        self.epoch = 0
        self._groups = None
        self._all = None
        self._eligible_count = None

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _cache_path(self) -> str:
        ds_list = getattr(self.dataset, "datasets", [])
        signature = {
            "length": len(self.dataset),
            "epoch": self.epoch,
            "seed": self.seed,
            "domain_id_min": self.domain_id_min,
            "domain_group_size": self.domain_group_size,
            "max_groups_per_batch": self.max_groups_per_batch,
            "allowed_styles": (None if self.allowed_styles is None
                               else sorted(self.allowed_styles)),
            "datasets": [str(getattr(d, "dataset_path", "")) for d in ds_list],
        }
        h = hashlib.sha1(repr(signature).encode()).hexdigest()[:16]
        root = os.environ.get("DACSC_CACHE_DIR", os.path.expanduser("~/.cache/dacsc"))
        return os.path.join(root, f"domain_sampler_{h}.pkl")

    def _build(self):
        cache = self._cache_path()
        if os.path.isfile(cache):
            with open(cache, "rb") as f:
                obj = pickle.load(f)
            self._groups = obj["groups"]
            self._all = obj["all"]
            self._eligible_count = len(self._all)
            print(f"[DomainBalancedSampler] cache hit: {cache}")
            return
        groups = defaultdict(list)
        all_indices = []
        for i in range(len(self.dataset)):
            ds, traj, _ = self.dataset.sample_step(i)
            mapping = getattr(ds, "domain_map", {})
            pair = mapping.get(int(traj), (None, None))
            style = pair[0]
            if style is None:
                # A style allowlist is a leakage barrier: never fall back to an
                # unlabeled sample that may belong to val/test.
                if self.allowed_styles is not None:
                    continue
                group = -1
            else:
                style = int(style)
                if (self.allowed_styles is not None
                        and style not in self.allowed_styles):
                    continue
                group = max(0, (style - self.domain_id_min) // self.domain_group_size)
            groups[group].append(i)
            all_indices.append(i)
        self._groups = {k: v for k, v in groups.items() if k >= 0 and v}
        # Derive the fill pool only from admissible groups. This makes it
        # impossible for unlabeled/out-of-split indices to enter a batch.
        self._all = [i for k in sorted(self._groups) for i in self._groups[k]]
        self._eligible_count = len(self._all)
        if not self._groups:
            raise RuntimeError("DomainBalancedSampler 没有找到可用的 domain 分组")
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tmp = cache + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"groups": self._groups, "all": self._all}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache)
        print(f"[DomainBalancedSampler] cache saved: {cache}")

    def __iter__(self):
        if self._groups is None:
            self._build()
        rng = random.Random(self.seed + self.epoch)
        group_ids_all = sorted(self._groups)
        n_batches = max(1, int(self._eligible_count or 0) // self.batch_size)
        for _ in range(n_batches):
            if (self.max_groups_per_batch is not None
                    and self.max_groups_per_batch < len(group_ids_all)):
                group_ids = rng.sample(group_ids_all, self.max_groups_per_batch)
            else:
                group_ids = group_ids_all
            out = []
            for g in group_ids:
                pool = self._groups[g]
                k = min(self.samples_per_domain, len(pool))
                out.extend(rng.sample(pool, k))
            # 用随机样本补齐到 batch_size
            while len(out) < self.batch_size:
                out.append(rng.choice(self._all))
            rng.shuffle(out)
            yield from out[:self.batch_size]

    def __len__(self):
        if self._groups is None:
            self._build()
        n_batches = max(1, int(self._eligible_count or 0) // self.batch_size)
        return n_batches * self.batch_size
