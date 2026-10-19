"""跨域激活分布统计 → 域敏感性 u^disc（方法文档 §6.4、§6.5 的左半支）。

做法：对每一层、每一个通道，按"外观域标签"（RoboCasa365 的 style_id）分别累计
激活的均值/方差，再算每个域的分布与"所有域合并后"的分布之间的 KL 散度。

    u^disc_{l,c} = (1/|D|) Σ_d KL( N(μ_d, σ_d²) || N(μ_*, σ_*²) )

为什么先做这个而不是域分类器梯度：
    1) 无监督、不依赖域分类器能否分得开；
    2) 不受"每域样本量不足"的影响（用 batch 内统计即可）；
    3) 方法文档 §6.3 规定：域分类器是第二支，且必须先证明准确率高于随机。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


def _top5(u: torch.Tensor, k: int = 5) -> list:
    """返回 u 中最大的 k 个 (层号, 通道号, 分数)。"""
    flat = u.flatten()
    idx = torch.topk(flat, k=min(k, flat.numel())).indices.tolist()
    C = u.shape[1]
    return [[int(i // C), int(i % C), round(float(flat[i]), 3)] for i in idx]


class DomainActivationStats:
    """按 (层, 域, 通道) 维护激活的均值/方差（EMA），并给出 u^disc。"""

    def __init__(self,
                 num_layers: int,
                 num_channels: int,
                 num_domains: int,
                 layer_indices: Optional[Sequence[int]] = None,
                 momentum: float = 0.99,
                 eps: float = 1e-6):
        self.layer_indices = list(layer_indices) if layer_indices else list(range(num_layers))
        self.num_channels = num_channels
        self.num_domains = num_domains
        self.momentum = momentum
        self.eps = eps
        L = len(self.layer_indices)
        # EMA 状态（CPU float64，保证长训练数值稳定；体量 L*D*C*2 = 12*50*384*2 ≈ 46 万，可忽略）
        self.mean = torch.zeros(L, num_domains, num_channels, dtype=torch.float64)
        self.var = torch.ones(L, num_domains, num_channels, dtype=torch.float64)
        self.count = torch.zeros(L, num_domains, dtype=torch.float64)     # EMA 权重
        self._seen = torch.zeros(L, num_domains, dtype=torch.bool)
        self._l2i = {l: i for i, l in enumerate(self.layer_indices)}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, layer_idx: int, feats: torch.Tensor, domain_ids: torch.Tensor) -> None:
        """累计一个 batch。

        feats      : (B, T, C) 该层 block 的**门控前**输出
        domain_ids : (B,) 外观域标签（style_id 的整数索引）
        """
        if layer_idx not in self._l2i:
            return
        li = self._l2i[layer_idx]
        # 注意：MPS（Apple GPU）不支持 float64，所以**在设备上算完归约后搬到 CPU**，
        # 累积用 CPU float64（体量 L*D*C ≈ 12*50*384 ≈ 23 万，可忽略）。
        x = feats.detach().to(torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b_mean = x.mean(dim=1).cpu().to(torch.float64)            # (B, C)  ← 先 cpu 再转
        b_var = x.var(dim=1, unbiased=False).cpu().to(torch.float64)
        d = domain_ids.detach().to(torch.long).view(-1).cpu()
        for dom in d.unique().tolist():
            m = d == dom
            if m.sum() == 0:
                continue
            cur_mean = b_mean[m].mean(dim=0)
            cur_var = b_var[m].mean(dim=0)
            if not self._seen[li, dom]:
                self.mean[li, dom] = cur_mean
                self.var[li, dom] = cur_var
                self._seen[li, dom] = True
            else:
                mo = self.momentum
                self.mean[li, dom] = mo * self.mean[li, dom] + (1 - mo) * cur_mean
                self.var[li, dom] = mo * self.var[li, dom] + (1 - mo) * cur_var
            self.count[li, dom] = self.momentum * self.count[li, dom] + (1 - self.momentum)

    # ------------------------------------------------------------------
    def u_disc(self, normalize: bool = True) -> torch.Tensor:
        """域敏感性分数 (L, C)。

        normalize=True  : 层内标准化后的分数（供门控正则使用，方法文档 §6.5）
        normalize=False : **原始 KL 值**（供诊断；注意它没有跨层可比性）
        """
        L = len(self.layer_indices)
        out = torch.zeros(L, self.num_channels, dtype=torch.float64)
        for li in range(L):
            seen = self._seen[li]
            if seen.sum() < 2:
                continue
            mu = self.mean[li][seen]                       # (D', C)
            var = self.var[li][seen].clamp_min(self.eps)
            w = self.count[li][seen].clamp_min(self.eps)   # (D',)
            w = w / w.sum()
            # 合并分布（全方差分解）
            mu_star = (w[:, None] * mu).sum(dim=0)                        # (C,)
            var_star = (w[:, None] * (var + mu ** 2)).sum(dim=0) - mu_star ** 2
            var_star = var_star.clamp_min(self.eps)
            # KL(N(mu,var) || N(mu*,var*))，对高斯有闭式解
            kl = (torch.log(var_star.sqrt() / var.sqrt())
                  + (var + (mu - mu_star) ** 2) / (2 * var_star)
                  - 0.5)                                              # (D', C)
            out[li] = (w[:, None] * kl).sum(dim=0)
        if not normalize:
            return out.to(torch.float32)
        # 层内标准化
        m = out.mean(dim=1, keepdim=True)
        s = out.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        return ((out - m) / s).to(torch.float32)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count,
                "seen": self._seen, "layer_indices": self.layer_indices,
                "num_channels": self.num_channels, "num_domains": self.num_domains,
                "momentum": self.momentum}

    def load_state_dict(self, sd: dict) -> None:
        self.mean = sd["mean"]; self.var = sd["var"]; self.count = sd["count"]
        self._seen = sd["seen"]
        self.layer_indices = sd["layer_indices"]; self._l2i = {l: i for i, l in enumerate(self.layer_indices)}
        self.num_channels = sd["num_channels"]; self.num_domains = sd["num_domains"]
        self.momentum = sd.get("momentum", self.momentum)

    def summary(self) -> dict:
        u = self.u_disc()
        n_seen = int(self._seen.sum())
        return {
            "n_layers_tracked": len(self.layer_indices),
            "n_domains_total": self.num_domains,
            "n_domain_layer_entries_seen": n_seen,
            "u_disc_mean": float(u.mean()),
            "u_disc_std": float(u.std()),
            "u_disc_top5_layer_channel": _top5(u),
        }


# ============================================================================
# 模式分发：`dgcd`（移植自 DgCD 官方实现）与 `kl_gauss`（我们自研的简化版）
#
# 为什么保留两套：
#   · `dgcd`     —— 有官方依据（CIKM 2024, MIT），作为**主用法**
#   · `kl_gauss` —— 我们早期的简化形式（逐通道高斯 KL），保留作**消融对照**，
#                   用来回答"打分方式本身有多大影响"
# ============================================================================
class DgCDDomainScorer:
    """用移植的 DgCD 打分，接口与 `DomainActivationStats` 对齐。

    ⚠️ 早期版本把每层激活**缓存到 CPU**，12 层 × (24,257,384) float32 ≈ **113 MB/步**，
    在 16GB 机器上会加剧换页（实测训练卡顿的元凶之一）。
    现改为：**在 update() 里就地算完分数，只保留 (层, 通道) 的结果**（~18 KB），不缓存激活。
    """

    def __init__(self, layer_indices, num_channels: int, num_domains: int,
                 rho: float = 1e-3, **_ignored):
        from dacsc.dg_methods import DgCDScorer
        self.layer_indices = list(layer_indices)
        self.num_channels = num_channels
        self.num_domains = num_domains
        self._scorer = DgCDScorer(rho=rho)
        self._score = torch.zeros(len(self.layer_indices), num_channels)
        self._seen = torch.zeros(len(self.layer_indices), dtype=torch.bool)
        self._l2i = {l: i for i, l in enumerate(self.layer_indices)}

    def update(self, layer_idx: int, feats: torch.Tensor, domain_ids: torch.Tensor) -> None:
        if layer_idx not in self._l2i:
            return
        did = domain_ids.detach().view(-1).to(torch.long).cpu()
        if did.unique().numel() < 2:
            return                                   # 单域 → 无差异
        x = feats.detach().to(torch.float32).cpu()   # 只在需要时拷一次，不缓存
        # ★ 必须显式开梯度：DgCD 打分内部要对自己的 scaler 求一次对抗梯度（Eq.7-8），
        #   若被 no_grad 包住，autograd.grad 会报 "does not require grad"。
        #   注意 x 已 detach，所以不会影响主干网络的计算图。
        with torch.enable_grad():
            s = self._scorer.score(x, did)           # (N, C)
        self._score[self._l2i[layer_idx]] = s.mean(dim=0).detach()
        self._seen[self._l2i[layer_idx]] = True
        del x, s

    def u_disc(self, normalize: bool = True) -> torch.Tensor:
        out = self._score.clone()
        if not normalize:
            return out
        m = out.mean(dim=1, keepdim=True)
        s = out.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (out - m) / s

    def state_dict(self) -> dict:
        return {"score": self._score.detach().cpu(),
                "seen": self._seen.detach().cpu(),
                "layer_indices": list(self.layer_indices),
                "num_channels": int(self.num_channels),
                "num_domains": int(self.num_domains)}

    def load_state_dict(self, sd: dict) -> None:
        for k in ("score", "seen"):
            if k in sd:
                setattr(self, "_" + ("score" if k == "score" else "seen"),
                        sd[k].detach().cpu())
        for k in ("layer_indices", "num_channels", "num_domains"):
            if k in sd:
                setattr(self, k, sd[k])
        self._l2i = {l: i for i, l in enumerate(self.layer_indices)}

    def summary(self) -> dict:
        u = self.u_disc()
        return {"mode": "dgcd",
                "n_layers_tracked": len(self.layer_indices),
                "n_layers_with_data": int(self._seen.sum()),
                "num_domains": self.num_domains,
                "u_disc_mean": float(u.mean()), "u_disc_std": float(u.std()),
                "u_disc_top5_layer_channel": _top5(u)}


def make_domain_scorer(mode: str, layer_indices, num_channels: int, num_domains: int,
                       momentum: float = 0.99, dgcd_rho: float = 1e-3):
    """按 mode 构造打分器。

    mode: "dgcd"     —— 移植自 DgCD 官方实现（推荐，主用法）
          "kl_gauss" —— 自研的逐通道高斯 KL（消融对照）
    """
    if mode == "dgcd":
        return DgCDDomainScorer(layer_indices, num_channels, num_domains, rho=dgcd_rho)
    if mode == "kl_gauss":
        s = DomainActivationStats(num_layers=max(layer_indices) + 1,
                                  num_channels=num_channels, num_domains=num_domains,
                                  layer_indices=layer_indices, momentum=momentum)
        return s
    raise ValueError(f"未知的域打分模式: {mode}")
