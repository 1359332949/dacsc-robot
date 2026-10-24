"""验证 token_style 真正接入了 CorpRoboCasa（patch tokens, pre-resampler）。"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import ROBOCASA_CORP_CONFIG, CorpRoboCasa  # noqa: E402
from dacsc.token_style import TokenStyleModulator  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    views = ROBOCASA_CORP_CONFIG["view_names"]
    B = 2
    images = {v: torch.randn(B, 3, 224, 224) for v in views}
    state = torch.randn(B, ROBOCASA_CORP_CONFIG["state_dim"])
    instruction = ["pick up the mug", "open the cabinet"]
    did = torch.tensor([0, 1])

    base = CorpRoboCasa()
    style = TokenStyleModulator(mode="cross_style_stats", alpha=1.0, dim=384)
    adapted = CorpRoboCasa(token_style=style)
    adapted.load_state_dict(base.state_dict(), strict=False)
    # cross_style_stats 是训练期增广：eval 下按设计关闭，因此在 train 模式验证。
    adapted.train()
    torch.manual_seed(123)
    rep_style = adapted.extract(images, state, instruction, domain_ids=did)
    adapted.token_style = None
    torch.manual_seed(123)
    rep_base = adapted.extract(images, state, instruction, domain_ids=did)

    assert rep_base.tokens.shape == rep_style.tokens.shape
    assert torch.isfinite(rep_style.tokens).all()
    assert not torch.allclose(rep_base.tokens, rep_style.tokens), "token_style had no effect"
    assert style.last_stats_t, "token_style stats not recorded"
    print("token_style in CoRP: ✓")
    print("  tokens", tuple(rep_style.tokens.shape))
    print("  stats", {k: round(float(v), 4) for k, v in style.last_stats_t.items()})

    # token_style=None 时保持官方路径
    assert base.token_style is None
    print("no-token_style path: ✓")
    print("CORP TOKEN STYLE TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
