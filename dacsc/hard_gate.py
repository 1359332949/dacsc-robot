"""v2：per-sample、随机单层、top-k hard mask + straight-through。

这是与旧 `ChannelGate`（全局 12×384 连续 sigmoid）并列的第二套机制：
  · 前向：每个样本选 top-k 通道丢弃，mask 为 0/1，并按 C/sum(mask) 重归一化
  · 反向：用 sigmoid 软 mask 的 straight-through（前向硬、反向软）
  · 每个训练 step 随机选 1 层；其它层恒等
  · 训练时启用，推理时可由 `train_only` 关闭
"""
from __future__ import annotations
import random
from typing import Dict, List, Sequence

import torch


class HardMaskController:
    def __init__(self, layer_indices: Sequence[int], drop_ratio: float = 0.33,
                 temperature: float = 0.1, train_only: bool = True):
        self.layer_indices = list(layer_indices)
        self.drop_ratio = float(drop_ratio)
        self.temperature = float(temperature)
        self.train_only = bool(train_only)
        self.layer: int | None = None
        self.mask: torch.Tensor | None = None
        # 当前 batch 的 mask 由 hook 内现场生成，避免“上一批 mask 套到下一批样本”。
        self.mask_builder = None
        self.build_enabled = True

    def set_mask_builder(self, fn) -> None:
        self.mask_builder = fn

    def select_layer(self) -> int | None:
        if not self.layer_indices:
            return None
        self.layer = int(random.choice(self.layer_indices))
        return self.layer

    def set_mask(self, mask: torch.Tensor | None) -> None:
        self.mask = mask

    def apply(self, x: torch.Tensor, layer_idx: int, training: bool) -> torch.Tensor:
        if self.layer != layer_idx or self.mask is None:
            return x
        if self.train_only and not training:
            return x
        m = self.mask.to(device=x.device, dtype=x.dtype)
        if m.shape[0] != x.shape[0]:
            m = m[:x.shape[0]]
        if m.shape[0] != x.shape[0]:
            return x
        return x * m[:, None, :]


def topk_ste_mask(scores: torch.Tensor, drop_ratio: float = 0.33,
                  temperature: float = 0.1) -> torch.Tensor:
    """scores: (N,C)。分数越高越应被丢弃。返回 (N,C) keep-mask。"""
    n, c = scores.shape
    k = max(0, min(int(round(float(drop_ratio) * c)), c))
    if k == 0:
        return torch.ones_like(scores)
    idx = torch.topk(scores, k, dim=1, largest=True).indices
    drop = torch.zeros_like(scores)
    drop.scatter_(1, idx, 1.0)
    hard_keep = 1.0 - drop
    # 阈值：第 k 个分数。低于阈值的通道更应保留。
    threshold = torch.topk(scores, k, dim=1, largest=True).values[:, -1:]
    soft_keep = torch.sigmoid((threshold - scores) / max(float(temperature), 1e-6))
    # straight-through：前向 hard_keep，反向走 soft_keep
    mask = hard_keep + soft_keep - soft_keep.detach()
    # 保持激活整体量级；避免 33% dropout 把特征整体压低
    return mask * c / mask.sum(dim=1, keepdim=True).clamp_min(1e-8)


def install_hard_gates(dino, controller: HardMaskController,
                       collect: Dict[str, List] | None = None,
                       retain_post_grad: bool = False,
                       retain_pre_grad: bool = False):
    blocks = getattr(dino, "blocks", None)
    assert blocks is not None, "该模型没有 .blocks"
    handles = []
    for i in range(len(blocks)):
        def hook(module, inputs, output, layer_idx=i):
            selected = controller.layer == layer_idx
            if selected and collect is not None:
                collect.setdefault("pre_gate", []).append((layer_idx, output.detach()))
                if retain_pre_grad and output.requires_grad:
                    output.retain_grad()
                    collect.setdefault("pre_gate_live", []).append((layer_idx, output))
            # 只用当前 batch 自己的特征生成本 batch 的 per-sample mask。
            if (selected and dino.training and controller.build_enabled
                    and controller.mask_builder is not None):
                mask = controller.mask_builder(output.detach(), layer_idx)
                controller.set_mask(mask)
            out = controller.apply(output, layer_idx, training=bool(dino.training))
            if selected and collect is not None:
                if retain_post_grad and out.requires_grad:
                    out.retain_grad()
                    collect.setdefault("post_gate_live", []).append((layer_idx, out))
                collect.setdefault("post_gate", []).append((layer_idx, out.detach()))
            return out
        handles.append(blocks[i].register_forward_hook(hook))
    return handles


def gumbel_ste_mask(scores: torch.Tensor, drop_ratio: float = 0.33,
                    temperature: float = 0.1, group_ids: torch.Tensor | None = None,
                    domain_shared: bool = False) -> torch.Tensor:
    """Gumbel-STE top-k：前向硬选择，反向走 soft 近似。

    domain_shared=True 时，同一粗域共享一份 Gumbel 噪声。这样随机性仍在，
    但不会把“同域应得到相近 mask”的结构打散。
    """
    n, c = scores.shape
    k = max(0, min(int(round(float(drop_ratio) * c)), c))
    if k == 0:
        return torch.ones_like(scores)
    if domain_shared and group_ids is not None:
        g = group_ids.detach().view(-1).to(torch.long)
        ng = int(g.max().item()) + 1 if g.numel() else 1
        u_group = torch.rand((ng, c), device=scores.device, dtype=scores.dtype)
        u = u_group[g].clamp_(1e-6, 1.0 - 1e-6)
    else:
        u = torch.rand_like(scores).clamp_(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(u))
    noisy = scores + float(temperature) * gumbel
    idx = torch.topk(noisy, k, dim=1, largest=True).indices
    drop = torch.zeros_like(scores)
    drop.scatter_(1, idx, 1.0)
    hard_keep = 1.0 - drop
    threshold = torch.topk(scores, k, dim=1, largest=True).values[:, -1:]
    soft_keep = torch.sigmoid((threshold - scores) / max(float(temperature), 1e-6))
    mask = hard_keep + soft_keep - soft_keep.detach()
    return mask * c / mask.sum(dim=1, keepdim=True).clamp_min(1e-8)


def mask_iou_by_domain(mask: torch.Tensor, domain_ids: torch.Tensor,
                       drop: bool = False):
    """返回 (same-domain mean IoU, different-domain mean IoU, gap)。

    drop=False 计算保留通道 IoU；drop=True 计算被丢弃通道 IoU。
    后者对 33% sparse mask 的域选择性更敏感。
    """
    keep = (mask.detach() <= 0) if drop else (mask.detach() > 0)
    same, diff = [], []
    n = keep.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            inter = (keep[i] & keep[j]).sum().item()
            union = (keep[i] | keep[j]).sum().item()
            if union == 0:
                continue
            v = inter / union
            (same if int(domain_ids[i]) == int(domain_ids[j]) else diff).append(v)
    s = sum(same) / len(same) if same else 0.0
    d = sum(diff) / len(diff) if diff else 0.0
    return s, d, s - d


def remove_hard_gates(handles) -> None:
    for h in handles:
        h.remove()


def score_separation_by_domain(scores: torch.Tensor, domain_ids: torch.Tensor):
    """域分数在通道空间中的同域/异域余弦相似度（越大越相似）。"""
    z = torch.nn.functional.normalize(scores.detach().float(), dim=1)
    same, diff = [], []
    n = z.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            v = float((z[i] * z[j]).sum())
            (same if int(domain_ids[i]) == int(domain_ids[j]) else diff).append(v)
    s = sum(same) / len(same) if same else 0.0
    d = sum(diff) / len(diff) if diff else 0.0
    return s, d, s - d
