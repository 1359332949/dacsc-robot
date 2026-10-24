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
    rep = m2.update_action_importance(return_summary=True)
    imp = style2.action_importance
    assert imp.numel() == 256, imp.shape
    assert float(imp.sum()) > 0
    assert rep is not None
    print("action_aware importance: ✓", tuple(imp.shape), "sum", round(float(imp.sum()), 4),
          "nonzero", int((imp > 0).sum()))
    # C1 regression: token_style must be registered when loaded via load_corp_robocasa
    import hashlib, json
    from dacsc.corp_robocasa import load_corp_robocasa
    sys.path.insert(0, str(ROOT / "repos" / "CoRP"))
    from corp.policy import CoRP
    from corp.preprocessing import Normalizer
    official_cfg = dict(ROBOCASA_CORP_CONFIG)
    official_cfg["view_names"] = ["agentview", "eye_in_hand"]
    official_cfg["action_dim"] = 7
    official_cfg["action_horizon"] = 12
    official_cfg["state_dim"] = 8
    nz = Normalizer({"format": "corp_libero_normalizer_v1",
                     "state_mean": [0.0]*8, "state_std": [1.0]*8,
                     "action_min": [-1.0]*7, "action_max": [1.0]*7})
    off = CoRP(official_cfg, nz)
    payload = {"format": "corp_libero_v2",
               "config_sha256": hashlib.sha256(json.dumps(official_cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "model": {k: v.detach().clone() for k, v in off.state_dict().items()},
               "normalizer": {"format": "corp_libero_normalizer_v1",
                              "state_mean": [0.0]*8, "state_std": [1.0]*8,
                              "action_min": [-1.0]*7, "action_max": [1.0]*7}}
    tmp = ROOT / "audit" / "corp_token_style_synth.pt"
    torch.save(payload, tmp)
    ts = TokenStyleModulator(mode="cross_style_stats", alpha=0.5, dim=384)
    loaded, _ = load_corp_robocasa(tmp, token_style=ts)
    assert loaded._style_handle is not None, "load_corp_robocasa did not register token_style hook"
    loaded.train()
    torch.manual_seed(7)
    loss = loaded.flow_matching_loss(images, state, instruction,
                                     torch.randn(2, ROBOCASA_CORP_CONFIG["action_horizon"], 12),
                                     domain_ids=did)
    assert ts.last_stats_t, "token_style did not run after load_corp_robocasa"
    tmp.unlink(missing_ok=True)
    print("load_corp_robocasa hook registration: ✓")

    print("CORP TOKEN STYLE TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
