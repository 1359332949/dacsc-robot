"""通道级软门控（DACSC-Robot 的核心模块）。

对应方法文档 §6.7：
    F_gate = g ⊙ F + (1 - g) ⊙ R_l(F, s_t)
v1 实现取 R = 恒等（**门控退化为软掩码**），理由见方法文档 §6.7：
    基线编码器已自带语言 FiLM，非平凡的 R 会与之撞车；先从最简形式验证机制。

插入位置：DINOv2 ViT-S/14 的 12 个 transformer block 的**输出**（每个 block 的隐藏维度 384）。
为什么是隐藏维度：CNN 里"通道"= 一张空间特征图；ViT 里与之对应的是 token 的隐藏维度
（一个隐藏维度在所有 token 上共享同一语义）。12 层 × 384 维 = 4608 个门控标量（每视角共享）。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn


class ChannelGate(nn.Module):
    """单层通道门控：g = σ(η)，输出 F ⊙ g。

    ⚠️ **已知的饱和问题（实测）**：初始 g≈0.999 时 η≈6.9，σ'(η)≈1e-3，
    所以"g 空间"的每步变化只有 lr × 1e-3。用主 lr(1e-4) 时 Δg/步 ≈ 1e-7，
    长训练也几乎不动。
    → 故 `gate_learning_rate` 必须显著高于主 lr（默认 1e-2，即 100×）。
    这一条是被训练烟测抓出来的，不是拍脑袋。
    """

    def __init__(self, num_channels: int, init_gate: float = 0.999):
        super().__init__()
        assert 0.0 < init_gate <= 1.0
        init_gate = min(init_gate, 1.0 - 1e-7)
        eta0 = math.log(init_gate / (1.0 - init_gate))
        self.num_channels = num_channels
        self.eta = nn.Parameter(torch.full((num_channels,), float(eta0)))

    @property
    def gates(self) -> torch.Tensor:
        """当前门控值 g ∈ (0,1)，形状 (C,)。"""
        return torch.sigmoid(self.eta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C) —— 广播到最后一维
        return x * self.gates.to(dtype=x.dtype, device=x.device)

    def extra_repr(self) -> str:
        with torch.no_grad():
            g = self.gates
        return (f"channels={self.num_channels}, "
                f"g_mean={g.mean():.4f}, g_min={g.min():.4f}, g_max={g.max():.4f}")


class DINOv2GateBank(nn.Module):
    """管理插在 DINOv2 各 block 输出上的门控集合。"""

    def __init__(self, num_layers: int, hidden_dim: int,
                 layer_indices: Sequence[int] | None = None,
                 init_gate: float = 0.999):
        super().__init__()
        idx = list(range(num_layers)) if layer_indices is None else list(layer_indices)
        for i in idx:
            assert 0 <= i < num_layers, f"层号越界: {i} (共 {num_layers} 层)"
        self.layer_indices = sorted(idx)
        self.gates = nn.ModuleDict({str(i): ChannelGate(hidden_dim, init_gate)
                                    for i in self.layer_indices})

    def __len__(self): return len(self.layer_indices)

    def gates_matrix(self) -> torch.Tensor:
        """所有层的门控值，形状 (L, C)。用于正则项与可视化。"""
        return torch.stack([self.gates[str(i)].gates for i in self.layer_indices], dim=0)

    def all_channels(self) -> int:
        return len(self.layer_indices) * next(iter(self.gates.values())).num_channels


def install_gates(dino: nn.Module,
                  bank: DINOv2GateBank,
                  collect: Dict[str, List] | None = None,
                  retain_post_grad: bool = False):
    """把门控以 forward hook 的形式装到 DINOv2 的 blocks 上（**不改 DINOv2 源码**）。

    同时可选收集每层 block 的**输出**（门控前），供域统计使用。

    返回 handles 列表，调用方负责在不需要时 handle.remove()。
    """
    blocks = getattr(dino, "blocks", None)
    assert blocks is not None, "该模型没有 .blocks，无法安装门控"
    handles = []
    for i in bank.layer_indices:
        gate = bank.gates[str(i)]

        def make_hook(layer_idx, gate_mod):
            def hook(module, inputs, output):
                if collect is not None:
                    collect.setdefault("pre_gate", []).append((layer_idx, output.detach()))
                out = gate_mod(output)             # 影响前向
                if collect is not None:
                    if retain_post_grad and out.requires_grad:
                        out.retain_grad()          # 供 backward 后读取 ∂L/∂f
                        collect.setdefault("post_gate_live", []).append((layer_idx, out))
                    collect.setdefault("post_gate", []).append((layer_idx, out.detach()))
                return out                          # hook 返回张量 → 替换输出
            return hook

        handles.append(blocks[i].register_forward_hook(make_hook(i, gate)))
    return handles


def remove_gates(handles: Iterable) -> None:
    for h in handles:
        h.remove()


def gate_summary(bank: DINOv2GateBank) -> dict:
    """门控状态的摘要，用于日志/断言。"""
    with torch.no_grad():
        g = bank.gates_matrix()
    return {
        "n_layers": len(bank),
        "n_channels_total": bank.all_channels(),
        "layer_indices": bank.layer_indices,
        "g_mean": float(g.mean()),
        "g_min": float(g.min()),
        "g_max": float(g.max()),
        "suppression_ratio": float(1.0 - g.mean()),   # 1 - 平均门控值
    }
