"""合成数据集：用于在本机（无 RoboCasa365 数据）跑通训练/门控管线。

设计要点：**外观必须真的依赖域标签**，否则 u^disc 没有东西可找、烟测就失去意义。
这里对每个域 d 注入：
  1) 一个域专属的颜色偏移（模拟"不同厨房的墙色/台面色"）
  2) 一个域专属的条纹图案（模拟"不同纹理"）
而**动作只依赖物体的位置**（与域无关）——这正是我们要验证的核心假设：
  域信息可以去掉，任务信息必须保住。

接口与官方数据集一致：__len__ / __getitem__ / get_normalizer / get_validation_dataset。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class SyntheticDomainDataset(BaseImageDataset):
    def __init__(self,
                 shape_meta: dict,
                 n_samples: int = 96,
                 n_domains: int = 4,
                 horizon: int = 10,
                 n_obs_steps: int = 2,
                 image_size: int = 256,
                 seed: int = 0,
                 domain_effect: float = 0.35,
                 action_domain_dependence: float = 0.0,
                 **kwargs):     # 官方 task 配置里还有 dataset_soup/pad_before/... 等参数，忽略之
        """
        domain_effect: 外观受域影响的强度（0 = 看不出域差异）
        action_domain_dependence: 动作是否也依赖域（正常应为 0；
                                  >0 时用来测试"动作信息被门控破坏"的情形）
        """
        self.shape_meta = shape_meta
        self.n_samples = int(n_samples)
        self.n_domains = int(n_domains)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.domain_effect = float(domain_effect)
        self.action_domain_dependence = float(action_domain_dependence)

        self.obs_meta = shape_meta["obs"]
        self.rgb_keys = [k for k, v in self.obs_meta.items() if v.get("type") == "rgb"]
        self.low_dim_keys = [k for k, v in self.obs_meta.items() if v.get("type") != "rgb"]
        self.action_dim = int(shape_meta["action"]["shape"][0])

        # 固定每个域的颜色偏移与条纹频率
        rng = np.random.RandomState(self.seed)
        self._dom_color = rng.uniform(-1, 1, size=(self.n_domains, 3)).astype(np.float32)
        self._dom_freq = rng.uniform(0.02, 0.09, size=(self.n_domains,)).astype(np.float32)
        self._dom_phase = rng.uniform(0, np.pi, size=(self.n_domains,)).astype(np.float32)

    # ------------------------------------------------------------------
    def __len__(self): return self.n_samples

    def _render(self, dom: int, sample_idx: int) -> np.ndarray:
        """生成一张 (H,W,3) 的合成"厨房照片"。"""
        rng = np.random.RandomState(self.seed * 100003 + sample_idx)
        H = W = self.image_size
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        # 基础场景：一块"桌面" + 一个"物体"方块（位置随机，与域无关）
        img = np.full((H, W, 3), 0.55, dtype=np.float32)
        obj_y = int(rng.uniform(0.3, 0.7) * H)
        obj_x = int(rng.uniform(0.3, 0.7) * W)
        s = H // 8
        img[obj_y:obj_y + s, obj_x:obj_x + s] = np.array([0.9, 0.35, 0.2], dtype=np.float32)
        # 域专属的条纹"纹理"
        stripe = 0.5 + 0.5 * np.sin(self._dom_freq[dom] * (xx + yy) + self._dom_phase[dom])
        img *= (1.0 - 0.25 * self.domain_effect * stripe)[..., None]
        # 域专属的颜色偏移
        img += self.domain_effect * self._dom_color[dom][None, None, :]
        return np.clip(img, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx) % self.n_samples
        dom = idx % self.n_domains
        rng = np.random.RandomState(self.seed + idx)
        obs = {}
        for k in self.rgb_keys:
            shape = tuple(self.obs_meta[k]["shape"])          # (3, H, W)
            frames = []
            for t in range(self.n_obs_steps):
                im = self._render(dom, idx * 100 + t)          # (H, W, 3)
                im = np.transpose(im, (2, 0, 1))               # → (3, H, W)
                frames.append(im)
            obs[k] = torch.from_numpy(np.stack(frames, 0).astype(np.float32))
        for k in self.low_dim_keys:
            d = int(self.obs_meta[k]["shape"][0])
            if k == "lang_emb":
                # 语言嵌入与域无关（模拟"指令只说任务，不说装修"）
                v = np.tile(np.linspace(-1, 1, d, dtype=np.float32), (self.n_obs_steps, 1))
            else:
                v = rng.uniform(-1, 1, size=(self.n_obs_steps, d)).astype(np.float32)
            obs[k] = torch.from_numpy(v)
        # 动作：只依赖"物体位置"（用 idx 派生），与域无关 → 正是我们要验证的性质
        base = np.array([np.sin(idx), np.cos(idx), 0.1 * idx], dtype=np.float32)
        act = np.tile(base, (self.horizon, 1))[:self.horizon, :3]
        act = np.concatenate([act,
                              rng.normal(0, 0.05, size=(self.horizon, self.action_dim - 3))
                              .astype(np.float32)], axis=1)
        if self.action_domain_dependence > 0:
            act[:, 0] += self.action_domain_dependence * dom
        return {"obs": obs,
                "action": torch.from_numpy(act.astype(np.float32)),
                "domain_id": torch.tensor(dom, dtype=torch.long)}

    # ------------------------------------------------------------------
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """与官方数据集保持一致的归一化结构（含图像键，否则 normalize 会 KeyError）。"""
        nz = LinearNormalizer()

        def ident(d):
            return SingleFieldLinearNormalizer.create_manual(
                scale=np.ones(d, dtype=np.float32),
                offset=np.zeros(d, dtype=np.float32),
                input_stats_dict={})

        for k, attr in self.obs_meta.items():
            if attr.get("type") == "rgb":
                nz[k] = get_image_range_normalizer()      # 官方对图像就是这么处理的
            else:
                nz[k] = ident(int(attr["shape"][0]))
        nz["action"] = ident(self.action_dim)
        return nz

    def get_validation_dataset(self):
        return SyntheticDomainDataset(
            shape_meta=self.shape_meta, n_samples=8, n_domains=self.n_domains,
            horizon=self.horizon, n_obs_steps=self.n_obs_steps,
            image_size=self.image_size, seed=self.seed + 1,
            domain_effect=self.domain_effect)
