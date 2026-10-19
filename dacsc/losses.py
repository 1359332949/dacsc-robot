"""门控正则与动作重要性（方法文档 §6.6、§6.8）。

L_gate = (1/LC) Σ_{l,c} [ λ_d · u_{l,c} · g_{l,c} + λ_a · v_{l,c} · (1-g_{l,c})² ]
         + λ_ρ · ( (1/LC) Σ (1 - g_{l,c}) - ρ )²

  · 第一项：域敏感性越高 → 越希望 g 小（抑制）
  · 第二项：动作重要性越高 → 越希望 g 接近 1（保护）
  · 第三项：整体抑制预算 ρ
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


def gate_regularizer(gates: torch.Tensor,
                     u_disc: torch.Tensor,
                     v_action: Optional[torch.Tensor] = None,
                     rho: float = 0.3,
                     lambda_d: float = 1.0,
                     lambda_a: float = 1.0,
                     lambda_rho: float = 1.0,
                     normalize_v: bool = True) -> Dict[str, torch.Tensor]:
    """计算 L_gate 及其分项。

    gates    : (L, C) 门控值 g ∈ (0,1)
    u_disc   : (L, C) 域敏感性（已层内标准化）
    v_action : (L, C) 动作重要性；None 时该项为 0（并返回 action_term=0）
    """
    assert gates.shape == u_disc.shape, f"形状不一致 {gates.shape} vs {u_disc.shape}"
    L, C = gates.shape
    u = u_disc.to(gates.dtype)

    dom_term = (u * gates).mean()

    if v_action is None:
        act_term = torch.zeros((), dtype=gates.dtype, device=gates.device)
    else:
        v = v_action.to(gates.dtype)
        if normalize_v:
            v = (v - v.mean()) / v.std().clamp_min(1e-6)
        act_term = (v * (1.0 - gates) ** 2).mean()

    budget_term = ((1.0 - gates).mean() - rho) ** 2
    total = lambda_d * dom_term + lambda_a * act_term + lambda_rho * budget_term
    return {
        "loss": total,
        "domain_term": dom_term.detach(),
        "action_term": act_term.detach(),
        "budget_term": budget_term.detach(),
        "suppression_ratio": (1.0 - gates).mean().detach(),
    }


def choose_rho(step: int, total_steps: int, rho_max: float, p: float = 1.0) -> float:
    """渐进抑制预算（方法文档 §7 阶段三，借鉴 PLACE 的渐进式思想）。

    ρ(t) = ρ_max · ( t / T_warmup )^p
    """
    if total_steps <= 0:
        return float(rho_max)
    frac = min(1.0, max(0.0, step / float(total_steps)))
    return float(rho_max) * (frac ** p)


class ActionImportanceEMA:
    """动作重要性 v^Taylor（方法文档 §6.6）。

    v_{l,c} = EMA( | f̄_{l,c} · ∂L_DP / ∂f̄_{l,c} | )
    实现方式：对每层 block 的**门控后**激活保留梯度（retain_grad），
    在 loss.backward() 之后读取 .grad，按 (层, 通道) 聚合。
    """

    def __init__(self, layer_indices, num_channels: int, momentum: float = 0.95):
        self.layer_indices = list(layer_indices)
        self.num_channels = num_channels
        self.momentum = momentum
        self._l2i = {l: i for i, l in enumerate(self.layer_indices)}
        self.value = torch.zeros(len(self.layer_indices), num_channels)
        self._seen = False

    @torch.no_grad()
    def update_from_tensors(self, tensors: Dict[int, torch.Tensor]) -> None:
        """tensors: {layer_idx: 门控后激活张量（已 retain_grad）}，需在 backward 之后调用。"""
        # hard gate 每步只选择一个层；未选层必须保持原值，不能当成 0 做 EMA。
        present = {}
        for l, t in tensors.items():
            if l not in self._l2i or t.grad is None:
                continue
            g = t.grad.detach().to(torch.float32)
            f = t.detach().to(torch.float32)
            if g.shape != f.shape:
                continue
            # |f · ∂L/∂f| 对 batch 与 token 维求均值 → (C,)
            score = (f * g).abs().mean(dim=tuple(range(f.dim() - 1)))
            present[self._l2i[l]] = score.cpu()
        if not present:
            return
        if self._seen:
            for li, score in present.items():
                self.value[li] = self.momentum * self.value[li] + (1 - self.momentum) * score
        else:
            self.value.zero_()
            for li, score in present.items():
                self.value[li] = score
            self._seen = True

    def state_dict(self) -> dict:
        return {"value": self.value.detach().cpu(),
                "seen": bool(self._seen),
                "momentum": float(self.momentum)}

    def load_state_dict(self, sd: dict) -> None:
        if "value" in sd:
            self.value = sd["value"].detach().cpu()
        if "seen" in sd:
            self._seen = bool(sd["seen"])
        if "momentum" in sd:
            self.momentum = float(sd["momentum"])

    def ready(self) -> bool:
        return self._seen

    def summary(self) -> dict:
        return {"ready": self._seen,
                "v_mean": float(self.value.mean()),
                "v_max": float(self.value.max())}


# ---------------------------------------------------------------------------
# 追加（2026-10-16）：按开源实现补充两个构件
# ---------------------------------------------------------------------------
class ActionImportanceRSC:
    """动作重要性：clean 分支 Taylor 分数 + RSC 式通道分位数归一化。

    原版 RSC 对**特征元素**做梯度分位数掩码；我们的门控单元是**通道**，
    因此必须先做通道级聚合，不能直接把“元素超过分位数的比例”再聚合成通道
    （那会让每个通道都接近 drop_fraction，数学上必然塌缩）。

    这里采用：
        raw_c = mean_{batch,token} |f_c * dL/df_c|       # Taylor importance
        v_c   = quantile_rank(raw_c)                     # RSC 式稳健归一化
    并只在**无 mask 的 clean forward**上更新，避免 hard mask 造成零梯度污染。
    """

    def __init__(self, layer_indices, num_channels: int,
                 drop_fraction: float = 0.3, momentum: float = 0.95):
        self.layer_indices = list(layer_indices)
        self.num_channels = num_channels
        self.drop_fraction = float(drop_fraction)
        self.momentum = float(momentum)
        self._l2i = {l: i for i, l in enumerate(self.layer_indices)}
        self.value = torch.zeros(len(self.layer_indices), num_channels)
        self.raw_value = torch.zeros(len(self.layer_indices), num_channels)
        self._seen = False

    @torch.no_grad()
    def update_from_tensors(self, tensors: Dict[int, torch.Tensor]) -> None:
        """tensors: {layer: clean pre-gate 激活（已 retain_grad）}。"""
        present, raw_present = {}, {}
        for l, t in tensors.items():
            if l not in self._l2i or t.grad is None:
                continue
            g = t.grad.detach().float()
            f = t.detach().float()
            if g.shape != f.shape or g.numel() == 0:
                continue
            score = (f * g).abs().mean(dim=tuple(range(f.dim() - 1)))
            if not torch.isfinite(score).all() or float(score.max()) <= 0:
                continue
            c = score.numel()
            order = torch.argsort(score)
            rank = torch.empty_like(score)
            # 映射到约 [-0.5, 0.5]，零均值、非零方差，跨 batch 尺度稳定。
            vals = torch.linspace(-0.5, 0.5, c, dtype=score.dtype, device=score.device)
            rank[order] = vals
            li = self._l2i[l]
            present[li] = rank.cpu()
            raw_present[li] = score.cpu()
        if not present:
            return
        if self._seen:
            for li, score in present.items():
                self.value[li] = self.momentum * self.value[li] + (1 - self.momentum) * score
                self.raw_value[li] = (self.momentum * self.raw_value[li]
                                      + (1 - self.momentum) * raw_present[li])
        else:
            self.value.zero_()
            self.raw_value.zero_()
            for li, score in present.items():
                self.value[li] = score
                self.raw_value[li] = raw_present[li]
            self._seen = True

    def state_dict(self) -> dict:
        return {"value": self.value.detach().cpu(),
                "raw_value": self.raw_value.detach().cpu(),
                "seen": bool(self._seen),
                "momentum": float(self.momentum),
                "drop_fraction": float(self.drop_fraction)}

    def load_state_dict(self, sd: dict) -> None:
        if "value" in sd:
            self.value = sd["value"].detach().cpu()
        if "raw_value" in sd:
            self.raw_value = sd["raw_value"].detach().cpu()
        if "seen" in sd:
            self._seen = bool(sd["seen"])
        if "momentum" in sd:
            self.momentum = float(sd["momentum"])
        if "drop_fraction" in sd:
            self.drop_fraction = float(sd["drop_fraction"])

    def ready(self) -> bool:
        return self._seen

    def summary(self) -> dict:
        raw = self.raw_value
        return {
            "ready": bool(self._seen),
            "v_mean": float(self.value.mean()),
            "v_std": float(self.value.std()) if self.value.numel() else 0.0,
            "v_min": float(self.value.min()) if self.value.numel() else 0.0,
            "v_max": float(self.value.max()) if self.value.numel() else 0.0,
            "raw_mean": float(raw.mean()),
            "raw_std": float(raw.std()) if raw.numel() else 0.0,
            "raw_rel_std": float(raw.std() / raw.mean().clamp_min(1e-12)) if raw.numel() else 0.0,
            "raw_nonzero": int((raw > 0).sum()),
        }

def coral_from_domains(features: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
    """按域分组后调用移植的 Deep CORAL（DomainBed, MIT）。

    features: (N, D)；domain_ids: (N,)
    """
    from dacsc.dg_methods import coral_loss
    d = domain_ids.detach().view(-1).to(torch.long)
    groups = [features[d == u] for u in d.unique(sorted=True)]
    return coral_loss(groups)
