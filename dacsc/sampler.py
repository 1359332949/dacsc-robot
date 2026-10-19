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
                 seed: int = 42):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.domain_id_min = int(domain_id_min)
        self.domain_group_size = max(int(domain_group_size), 1)
        self.samples_per_domain = int(samples_per_domain)
        self.seed = int(seed)
        self.epoch = 0
        self._groups = None
        self._all = None
        self._n_batches = max(1, len(dataset) // self.batch_size)

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
                group = -1
            else:
                group = max(0, (int(style) - self.domain_id_min) // self.domain_group_size)
            groups[group].append(i)
            all_indices.append(i)
        self._groups = {k: v for k, v in groups.items() if k >= 0 and v}
        self._all = all_indices
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
        group_ids = sorted(self._groups)
        for _ in range(self._n_batches):
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
        return self._n_batches * self.batch_size
