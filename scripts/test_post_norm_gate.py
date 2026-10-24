"""验证 gate_position=post_norm_tokens 的接线与 forward（CPU）。"""
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
        enable_gating=True, gate_mode="hard", gate_position="post_norm_tokens",
        hard_train_only=False, hard_domain_mode="random",
        use_action_importance=False,
    )
    # 接线检查
    assert p.gate_position == "post_norm_tokens"
    assert getattr(p.obs_encoder, "hard_controller", None) is p.hard_controller
    assert p.obs_encoder.gate_position == "post_norm_tokens"
    assert p._gate_handles == []
    print("post_norm wiring: ✓ (no block hooks, controller attached to encoder)")

    p.train()
    B = 2
    obs = dummy_obs(cfg.shape_meta, B=B, To=int(cfg.n_obs_steps), seed=1)
    batch = {
        "obs": obs,
        "action": torch.rand(B, int(cfg.horizon), cfg.shape_meta.action.shape[0]),
        "domain_id": torch.randint(0, 4, (B,)),
        "layout_id": torch.randint(0, 4, (B,)),
    }
    loss = p.compute_loss(batch)
    assert torch.isfinite(loss), loss
    loss.backward()
    rep = p.gate_report()
    layer = rep["metrics"].get("layer")
    print(f"post_norm train loss={float(loss):.4f} layer={layer} ✓")
    assert layer == -1, layer

    p.eval()
    with torch.no_grad():
        out = p.predict_action(dummy_obs(cfg.shape_meta, B=1, To=int(cfg.n_obs_steps), seed=2))
    print("post_norm eval predict_action:", tuple(out["action"].shape), "✓")
    print("POST NORM GATE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
