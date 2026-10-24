"""验证 token_style 真正接入 CorpRoboCasa，且 action-aware 的 importance 会更新。"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import ROBOCASA_CORP_CONFIG, CorpRoboCasa  # noqa: E402
from dacsc.token_style import TokenStyleModulator  # noqa: E402


def _inputs(b=2):
    views = ROBOCASA_CORP_CONFIG["view_names"]
    return (
        {v: torch.randn(b, 3, 224, 224) for v in views},
        torch.randn(b, ROBOCASA_CORP_CONFIG["state_dim"]),
        ["pick up the mug", "open the cabinet"][:b],
        torch.arange(b),
    )


def main() -> int:
    torch.manual_seed(0)
    images, state, instruction, did = _inputs()

    base = CorpRoboCasa()
    style = TokenStyleModulator(mode="cross_style_stats", alpha=1.0, dim=384)
    adapted = CorpRoboCasa(token_style=style)
    adapted.load_state_dict(base.state_dict(), strict=False)

    base.train()
    adapted.train()
    torch.manual_seed(123)
    rep_base = base.extract(images, state, instruction, domain_ids=did)
    torch.manual_seed(123)
    rep_style = adapted.extract(images, state, instruction, domain_ids=did)
    assert rep_base.tokens.shape == rep_style.tokens.shape
    assert torch.isfinite(rep_style.tokens).all()
    assert not torch.allclose(rep_base.tokens, rep_style.tokens), "token_style had no effect"
    assert style.last_stats_t, "token_style stats not recorded"
    print("token_style hook: ✓", tuple(rep_style.tokens.shape),
          {k: round(float(v), 4) for k, v in style.last_stats_t.items()})

    # action-aware：backward 后 importance 必须变成 (T,) 且非零
    style2 = TokenStyleModulator(mode="cross_style_stats_action_aware", alpha=0.5, dim=384)
    m2 = CorpRoboCasa(token_style=style2)
    m2.load_state_dict(base.state_dict(), strict=False)
    m2.train()
    action = torch.randn(2, ROBOCASA_CORP_CONFIG["action_horizon"], 12)
    loss = m2.flow_matching_loss(images, state, instruction, action, domain_ids=did)
    loss.backward()
    rep = m2.update_action_importance()
    imp = style2.action_importance
    assert imp.numel() == 256, imp.shape
    assert float(imp.sum()) > 0
    assert rep is not None
    print("action_aware importance: ✓", tuple(imp.shape), "sum", round(float(imp.sum()), 4),
          "nonzero", int((imp > 0).sum()))
    print("CORP TOKEN STYLE TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
