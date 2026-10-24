"""回归测试：train_only=False 时 eval 必须用当前观测重建 mask，不能沿用训练 stale mask。"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from test_gating import build_policy, dummy_obs  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    p, cfg = build_policy(
        enable_gating=True, gate_mode="hard", hard_train_only=False,
        hard_domain_mode="random", use_action_importance=False,
        hard_layer_indices=[0], gate_position="block_pre_norm",
    )
    B = 2
    action = torch.randn(B, int(cfg.horizon), cfg.shape_meta.action.shape[0])
    did = torch.tensor([0, 1])

    p.train()
    p.compute_loss({"obs": dummy_obs(cfg.shape_meta, B=B, To=int(cfg.n_obs_steps), seed=1),
                    "action": action, "domain_id": did})
    m_train = p.hard_controller.mask.clone()

    p.eval()
    with torch.no_grad():
        p.compute_loss({"obs": dummy_obs(cfg.shape_meta, B=B, To=int(cfg.n_obs_steps), seed=2),
                        "action": action, "domain_id": did})
    m_eval = p.hard_controller.mask.clone()

    assert m_eval is not None
    assert not torch.equal(m_train, m_eval), "eval used stale training mask"
    print("stale-mask regression: ✓ (eval rebuilt mask from current observation)")

    # train_only=True: eval 不重建，保持训练 mask（这是正确行为）
    p2, cfg2 = build_policy(
        enable_gating=True, gate_mode="hard", hard_train_only=True,
        hard_domain_mode="random", use_action_importance=False, hard_layer_indices=[0],
    )
    p2.train()
    p2.compute_loss({"obs": dummy_obs(cfg2.shape_meta, B=B, To=int(cfg2.n_obs_steps), seed=1),
                     "action": action, "domain_id": did})
    m1 = p2.hard_controller.mask.clone()
    p2.eval()
    with torch.no_grad():
        p2.compute_loss({"obs": dummy_obs(cfg2.shape_meta, B=B, To=int(cfg2.n_obs_steps), seed=2),
                         "action": action, "domain_id": did})
    assert torch.equal(m1, p2.hard_controller.mask)
    print("train_only=True: ✓ (eval 不重建、gate 关闭)")
    print("STALE MASK TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
