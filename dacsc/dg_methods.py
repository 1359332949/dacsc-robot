"""从开源域泛化代码库移植的构件（**不自己造轮子**）。

出处与许可
----------
1. `CORAL`            ← facebookresearch/DomainBed, `domainbed/algorithms.py` 的
                        `AbstractMMD.mmd()` 的 mean_cov 分支。许可 **MIT**。
2. `rsc_mask`         ← 同文件 `class RSC`。许可 **MIT**。
3. `grad_reverse`     ← DomainBed `AbstractDANN` 的 `ReverseLayerF`。许可 **MIT**。
4. `DgCDScorer`       ← gyeomo/DgCD, `domainbed/networks/DgCD.py`（CIKM 2024 官方实现）。许可 **MIT**。

移植时做的必要改动（都在 docstring 里标明）：
  · 去掉硬编码 `.cuda()` / `device="cuda"` → 设备无关（本机跑 Apple MPS）
  · DgCD 的"环境"按**域标签**切分，而不是硬编码每 32 个样本一块
  · 输入张量形状适配我们的 (N, T, C)（ViT token 序列）
"""

from __future__ import annotations

import warnings

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1) Deep CORAL —— 逐对域的二阶统计对齐
#    出处：DomainBed `AbstractMMD.mmd()` 的 mean_cov 分支（MIT）
# ---------------------------------------------------------------------------
def coral_loss(features_by_domain: Sequence[torch.Tensor],
               unbiased: bool = True) -> torch.Tensor:
    """域间均值+协方差差异（Deep CORAL 的损失项）。

    features_by_domain: 每个元素是一个域的 (n_d, D) 特征
    """
    feats = [f for f in features_by_domain if f is not None and f.shape[0] >= 2]
    nmb = len(feats)
    if nmb < 2:
        return torch.zeros((), device=features_by_domain[0].device
                           if features_by_domain else "cpu")
    penalty = 0.0
    for i in range(nmb):
        x = feats[i]
        for j in range(i + 1, nmb):
            y = feats[j]
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            denom_x = (len(x) - 1) if unbiased else len(x)
            denom_y = (len(y) - 1) if unbiased else len(y)
            cova_x = (cent_x.t() @ cent_x) / max(denom_x, 1)
            cova_y = (cent_y.t() @ cent_y) / max(denom_y, 1)
            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()
            penalty = penalty + mean_diff + cova_diff
    return penalty / (nmb * (nmb - 1) / 2)


# ---------------------------------------------------------------------------
# 2) RSC —— 梯度分位数掩码（我们 v^Taylor 的同族做法）
#    出处：DomainBed `class RSC`（MIT）
# ---------------------------------------------------------------------------
def rsc_grad_mask(features: torch.Tensor,
                  logits: torch.Tensor,
                  labels_onehot: torch.Tensor,
                  drop_fraction: float = 0.3) -> torch.Tensor:
    """按"对正确类别的梯度"的分位数生成掩码（保留低梯度通道）。

    features: (N, D)；logits: (N, K)；labels_onehot: (N, K)
    返回 mask: (N, D)，1 = 保留。

    ⚠️ 前置条件：`logits` 必须是 `features` 的函数（有计算图），
    否则 `autograd.grad` 会报 "does not require grad"。
    """
    import numpy as np
    from torch import autograd
    if not logits.requires_grad and (logits.grad_fn is None):
        raise ValueError("logits 必须由 features 计算得到（需要计算图）")
    # Eq.(1) 对表示求梯度
    g = autograd.grad((logits * labels_onehot).sum(), features, retain_graph=True)[0]
    # Eq.(2) 取分位数
    pct = np.percentile(g.detach().cpu().numpy(), (1 - drop_fraction) * 100, axis=1)
    pct = torch.as_tensor(pct, dtype=g.dtype, device=g.device).unsqueeze(1) \
        .repeat(1, g.size(1))
    return g.lt(pct).float()


# ---------------------------------------------------------------------------
# 3) 梯度反转层（DANN / DomainDrop 都用）
#    出处：DomainBed `AbstractDANN.ReverseLayerF`（MIT）
# ---------------------------------------------------------------------------
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


def dann_loss(features: torch.Tensor, domain_labels: torch.Tensor, num_domains: int,
              head: nn.Module, lambd: float = 1.0) -> torch.Tensor:
    """DANN 的域交叉熵（特征先过 GRL，故梯度反向）。"""
    return F.cross_entropy(head(grad_reverse(features, lambd)), domain_labels)


# ---------------------------------------------------------------------------
# 4) DgCD —— 分布差异引导的通道打分（★ 我们的 u^disc 就用它）
#    出处：gyeomo/DgCD `domainbed/networks/DgCD.py`（CIKM 2024 官方, MIT）
# ---------------------------------------------------------------------------
class DgCDScorer(nn.Module):
    """DgCD 的通道打分与掩码生成。

    ⚠️ 关键：**这不是"逐通道高斯 KL"**（我们早期文档写错了），而是：
        1. GAP 得到 (N, C) 的通道向量
        2. 按"环境"（= 我们的**域**）与全体分别标准化，再乘可学习仿射 Scaling
        3. 对**通道维**做 log_softmax → 把通道向量当成一个分布
        4. 用 KL 比较"全体"与"各环境"分布，**反向传播并对 scaler 做梯度上升**
           （对抗式地拉大差异）
        5. 用更新后的 scaler 重算对齐；`scores = (总体 − 环境)²`（平方差，非 KL）
        6. 加权随机采样（WRS）生成掩码，并**重归一化保持量级**

    相对原实现的改动：
      · 设备无关（原代码硬编码 .cuda()）
      · 环境按 `domain_ids` 切分（原代码硬编码每 32 样本一块）
    """

    def __init__(self, rho: float = 1e-3):
        super().__init__()
        self.rho = float(rho)                      # 对抗步长（DgCD 的 ρ）
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    # -- 内部工具 ---------------------------------------------------------
    class _Scaling(nn.Module):
        """可学习仿射 x*γ+β（DgCD 的 Eq.5）。"""
        def __init__(self, n: int, c: int, device, dtype):
            super().__init__()
            self.mean = nn.Parameter(torch.zeros((n, c), device=device, dtype=dtype))
            self.var = nn.Parameter(torch.ones((n, c), device=device, dtype=dtype))

        def forward(self, x):
            return x * self.var + self.mean

    @staticmethod
    def _minmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        x_max = x.max(dim=dim, keepdim=True)[0]
        x_min = x.min(dim=dim, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min).clamp_min(1e-12)

    def _alignment(self, x, scaler_t, env_slices, scaler_e):
        """DgCD 的 Eq.(4)(5)：先标准化再仿射。"""
        env_wise = []
        for k, sl in enumerate(env_slices):
            s = x[sl].clone()
            s = (s - s.mean(dim=0, keepdim=True)) / torch.sqrt(s.var(dim=0, keepdim=True) + 1e-5)
            env_wise.append(scaler_e[k](s))
        env_wise = torch.cat(env_wise, dim=0)
        total = (x - x.mean(dim=0, keepdim=True)) / torch.sqrt(x.var(dim=0, keepdim=True) + 1e-5)
        return env_wise, scaler_t(total)

    def _make_mask(self, scores: torch.Tensor, ratio: float) -> torch.Tensor:
        """DgCD Eq.(12)(13)：加权随机采样 + 重归一化。"""
        n, c = scores.shape
        r = torch.rand(scores.shape, device=scores.device, dtype=scores.dtype)
        key = r.pow(1.0 / scores.clamp_min(1e-8))
        k = min(int(ratio * c), c - 1)
        threshold = torch.sort(key, dim=1, descending=True)[0][:, k]
        mask = 1.0 - (key > threshold.view(n, 1)).to(scores.dtype)
        mask = mask.view(n, c, 1, 1)
        # ★ 重归一化：保持整体激活量级（原实现里容易漏掉的一步）
        return mask * mask.numel() / mask.sum().clamp_min(1e-8)

    # -- 主入口 -----------------------------------------------------------
    def score(self, feats: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """返回逐 (样本, 通道) 的域差异分数 (N, C)。

        feats: (N, T, C) 或 (N, C)
        """
        x = feats if feats.dim() == 2 else feats.mean(dim=1)
        x = x.detach()
        n, c = x.shape
        device, dtype = x.device, x.dtype

        # 按域切分成"环境"
        d = domain_ids.detach().view(-1).to(torch.long)
        uniq = d.unique(sorted=True)
        env_slices = [torch.nonzero(d == u, as_tuple=False).view(-1) for u in uniq]
        env_slices = [sl for sl in env_slices if sl.numel() >= 2]
        if len(env_slices) < 2:
            warnings.warn(
                f"DgCDScorer.score: only {len(env_slices)} domain(s) with >=2 "
                f"samples (unique={len(uniq)}); returning zeros. "
                "Use a domain-balanced sampler so each domain has >=2 samples.",
                RuntimeWarning, stacklevel=2)
            return torch.zeros_like(x)

        # score() 会在 DINOv2 forward hook 内被调用；外层可能处于 no_grad
        # （推理/冻结分支），但 DgCD 的 scaler 对抗步必须显式开启 autograd。
        with torch.enable_grad():
            scaler_t = self._Scaling(n, c, device, dtype)
            scaler_e = [self._Scaling(sl.numel(), c, device, dtype) for sl in env_slices]

            env_w, tot = self._alignment(x, scaler_t, env_slices, scaler_e)
            dist = self.kl_loss(F.log_softmax(tot, dim=1), F.log_softmax(env_w, dim=1))
            # ★ 对抗步：对 scaler 做梯度上升（DgCD Eq.7-8），不更新主干
            grads = torch.autograd.grad(dist, list(scaler_t.parameters()) +
                                        [p for s in scaler_e for p in s.parameters()],
                                        retain_graph=False, allow_unused=True)
            for p, g in zip(list(scaler_t.parameters()) +
                            [p for s in scaler_e for p in s.parameters()], grads):
                if g is None:
                    continue
                norm = torch.sqrt(torch.norm(g, dim=-1, keepdim=True) + 1e-12)
                p.data = p.data + self.rho * g.div(norm)

            env_w, tot = self._alignment(x, scaler_t, env_slices, scaler_e)
            gram = torch.mm(tot.t() + 1e-7, env_w + 1e-7).diag()
            tot = self._minmax(tot / gram.clamp_min(1e-12))
            env_w = self._minmax(env_w / gram.clamp_min(1e-12))
            out = self._minmax((tot - env_w).pow(2))
        return out.detach()


__all__ = ["coral_loss", "rsc_grad_mask", "grad_reverse", "dann_loss", "DgCDScorer"]
