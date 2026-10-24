"""gate_position / train_only 语义单元测试（CPU）。"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dacsc.hard_gate import HardMaskController  # noqa: E402


def _ctrl(train_only: bool) -> HardMaskController:
    c = HardMaskController([-1], drop_ratio=0.5, train_only=train_only)
    c.layer = -1
    c.set_mask(torch.tensor([[0.0, 1.0, 1.0, 1.0]]))
    return c


def main() -> int:
    x = torch.ones(1, 2, 4)
    keep = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])

    # train_only=True: 训练期正则，eval 关闭
    c = _ctrl(True)
    assert torch.equal(c.apply(x, -1, training=True), x * keep)
    assert torch.equal(c.apply(x, -1, training=False), x)
    print("train_only=True  : train 生效 / eval 关闭  ✓")

    # train_only=False: 推理期通道选择，train/eval 都生效
    c = _ctrl(False)
    assert torch.equal(c.apply(x, -1, training=True), x * keep)
    assert torch.equal(c.apply(x, -1, training=False), x * keep)
    print("train_only=False : train/eval 都生效（推理期选择）✓")

    # layer 不匹配时不干预
    c = _ctrl(False)
    assert torch.equal(c.apply(x, 0, training=True), x)
    print("layer mismatch   : 不干预  ✓")
    print("GATE POSITION SEMANTICS OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
