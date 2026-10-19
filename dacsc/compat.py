"""兼容 shim：官方 DP fork pin 的是 diffusers==0.11.1，新版把一些 typing 符号移走了。

**不改动官方仓库源码**——只在内存里补齐缺失的符号。
正式训练环境建议按官方 conda_environment.yaml 装 diffusers==0.11.1，届时本 shim 自动空转。
"""
from __future__ import annotations
import typing

_APPLIED: list[str] = []


def apply() -> list[str]:
    """补齐缺失符号，返回实际打上的补丁列表（幂等）。"""
    global _APPLIED
    try:
        import diffusers.optimization as o
    except Exception:
        return _APPLIED
    try:
        from torch.optim import Optimizer
    except Exception:
        Optimizer = None
    for name, obj in (("Union", typing.Union),
                      ("Optional", typing.Optional),
                      ("Optimizer", Optimizer)):
        if obj is not None and not hasattr(o, name):
            setattr(o, name, obj)
            _APPLIED.append(f"diffusers.optimization.{name}")
    return _APPLIED


applied = apply()
