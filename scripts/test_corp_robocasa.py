"""CoRP-RoboCasa 适配器 smoke：随机权重下验证结构/形状/训练路径/权重合并。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from dacsc.corp_robocasa import (  # noqa: E402
    ROBOCASA_CORP_CONFIG,
    CorpRoboCasa,
    load_corp_robocasa,
    VIEW_TO_OFFICIAL,
)

B = 2
H = ROBOCASA_CORP_CONFIG["action_horizon"]
A = ROBOCASA_CORP_CONFIG["action_dim"]
S = ROBOCASA_CORP_CONFIG["state_dim"]
VIEWS = ROBOCASA_CORP_CONFIG["view_names"]


def make_inputs(device="cpu"):
    images = {v: torch.randn(B, 3, 224, 224, device=device) for v in VIEWS}
    state = torch.randn(B, S, device=device)
    instruction = ["pick up the mug", "open the cabinet"]
    action = torch.randn(B, H, A, device=device)
    return images, state, instruction, action


def main() -> int:
    model = CorpRoboCasa()
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[1] build ok, params={n_params:,}, views={VIEWS}, "
          f"state={S}, action={A}, horizon={H}")

    images, state, instruction, action = make_inputs()
    rep = model.extract(images, state, instruction)
    print(f"[2] extract ok: tokens={tuple(rep.tokens.shape)}, "
          f"pooled_language={tuple(rep.pooled_language.shape)}, "
          f"dtype={rep.action_dtype}")

    loss = model.flow_matching_loss(images, state, instruction, action)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    print(f"[3] flow_matching_loss ok: {float(loss):.4f}")
    loss.backward()
    g = next(p.grad for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"[4] backward ok: grad sample {tuple(g.shape)}")

    model.eval()
    with torch.no_grad():
        act = model.sample_action(images, state, instruction)
    assert tuple(act.shape) == (B, H, A), act.shape
    print(f"[5] sample_action ok: {tuple(act.shape)}")

    # ---- 权重合并逻辑测试（合成官方 2-view checkpoint）----
    print("[6] weight-merge logic with synthetic official checkpoint ...")
    sys.path.insert(0, str(_REPO_ROOT / "repos" / "CoRP"))
    from corp.policy import CoRP
    from corp.preprocessing import Normalizer

    official_cfg = dict(ROBOCASA_CORP_CONFIG)
    official_cfg["view_names"] = ["agentview", "eye_in_hand"]
    official_cfg["action_dim"] = 7
    official_cfg["action_horizon"] = 12
    official_cfg["state_dim"] = 8

    normalizer = Normalizer({
        "format": "corp_libero_normalizer_v1",
        "state_mean": [0.0] * 8,
        "state_std": [1.0] * 8,
        "action_min": [-1.0] * 7,
        "action_max": [1.0] * 7,
    })
    official = CoRP(official_cfg, normalizer)
    official_sd = {k: v.detach().clone() for k, v in official.state_dict().items()}

    import hashlib, json
    fake_cfg = official_cfg
    payload = {
        "format": "corp_libero_v2",
        "config_sha256": hashlib.sha256(
            json.dumps(fake_cfg, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "model": official_sd,
        "normalizer": {
            "format": "corp_libero_normalizer_v1",
            "state_mean": [0.0] * 8,
            "state_std": [1.0] * 8,
            "action_min": [-1.0] * 7,
            "action_max": [1.0] * 7,
        },
    }
    tmp = _REPO_ROOT / "audit" / "corp_robocasa_synthetic_ckpt.pt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, tmp)

    adapted, report = load_corp_robocasa(tmp)
    print(f"    total={report['total']} copied={report['copied']} "
          f"random_init={report['random_init']}")
    # 断言：共享层已拷贝；接口层随机；view query 做了部分迁移。
    assert report["copied"] > 0
    assert report["partial_keys"] == [
        "representation_extractor.cross_attention_resampler.queries",
        "representation_extractor.cross_attention_resampler.view_embedding",
    ]
    q = adapted.representation_extractor.cross_attention_resampler.queries
    ve = adapted.representation_extractor.cross_attention_resampler.view_embedding
    # view0(agentview_left)<-agentview, view1(eye_in_hand)<-eye_in_hand 已拷贝；
    # view2(agentview_right) 随机（不等于官方任一 view）。
    assert torch.equal(q[0], official_sd["representation_extractor.cross_attention_resampler.queries"][0])
    assert torch.equal(q[1], official_sd["representation_extractor.cross_attention_resampler.queries"][1])
    assert not torch.equal(q[2], official_sd["representation_extractor.cross_attention_resampler.queries"][0])
    assert torch.equal(ve[0], official_sd["representation_extractor.cross_attention_resampler.view_embedding"][0])
    print("    view-param partial transfer verified (view0/1 copied, view2 random)")

    # state/action 投影层应随机初始化（形状不匹配，不能拷贝）
    sp = adapted.representation_extractor.state_encoder.projection.weight  # [768,9]
    ap = adapted.action_generator.action_projection.weight               # [768,12]
    assert tuple(sp.shape) == (768, 9)
    assert tuple(ap.shape) == (768, 12)
    print(f"    interface projections random-init verified: state {tuple(sp.shape)}, "
          f"action {tuple(ap.shape)}")

    # 清理
    tmp.unlink(missing_ok=True)
    print("ALL SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
