"""真实 corp_libero.pt 权重下验证 CoRP-RoboCasa 适配器（服务器 GPU）。

检查：
  1. checkpoint format / config_sha256
  2. 官方 2-view CoRP 能 strict=True 加载
  3. 适配器 load_corp_robocasa 合并报告
  4. 共享层权重与官方逐 tensor 一致
  5. GPU fp32/bf16 前向、flow loss、sample
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from dacsc.corp_robocasa import (  # noqa: E402
    ROBOCASA_CORP_CONFIG,
    CorpRoboCasa,
    load_corp_robocasa,
)

CKPT = Path("/hy-tmp/models/corp/CoRP/checkpoints/corp_libero.pt")
CFG = Path("/hy-tmp/models/corp/CoRP/checkpoints/config.json")
OUT = Path("/hy-tmp/runs/corp_robocasa_p1")
OUT.mkdir(parents=True, exist_ok=True)


def config_digest(config: dict) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    report: dict = {}
    # 1. checkpoint contract
    payload = torch.load(CKPT, map_location="cpu", weights_only=True)
    off_cfg = json.loads(CFG.read_text())
    assert payload["format"] == "corp_libero_v2"
    assert payload["config_sha256"] == config_digest(off_cfg), "config sha mismatch"
    report["checkpoint"] = {
        "path": str(CKPT),
        "sha256": hashlib.sha256(CKPT.read_bytes()).hexdigest(),
        "format": payload["format"],
        "config_sha256_ok": True,
        "n_tensors": len(payload["model"]),
    }
    print("[1] checkpoint contract ok:", report["checkpoint"]["n_tensors"], "tensors")

    # 2. official strict load
    from corp.checkpoint import load_policy
    official = load_policy(CKPT, device="cpu")
    print("[2] official CoRP strict load ok")

    # 3. adapter merge
    adapted, merge = load_corp_robocasa(CKPT, device="cpu")
    report["merge"] = {
        k: merge[k] for k in ("total", "copied", "random_init")
    }
    report["merge"]["partial_keys"] = merge["partial_keys"]
    report["merge"]["random_init_keys"] = merge["random_init_keys"]
    print(f"[3] adapter merge: copied={merge['copied']} "
          f"random={merge['random_init']} partial={merge['partial_keys']}")

    # 4. shared-layer equality
    off_sd = official.state_dict()
    ad_sd = adapted.state_dict()
    shared_mismatch = []
    n_shared = 0
    for k, v in off_sd.items():
        if k in ad_sd and tuple(ad_sd[k].shape) == tuple(v.shape):
            n_shared += 1
            if not torch.equal(v, ad_sd[k]):
                shared_mismatch.append(k)
    assert not shared_mismatch, f"shared mismatch: {shared_mismatch[:5]}"
    report["shared_tensors_verified"] = n_shared
    print(f"[4] shared tensors identical: {n_shared}")

    # 5. GPU forward / loss / sample
    device = "cuda" if torch.cuda.is_available() else "cpu"
    views = ROBOCASA_CORP_CONFIG["view_names"]
    H = ROBOCASA_CORP_CONFIG["action_horizon"]
    A = ROBOCASA_CORP_CONFIG["action_dim"]
    S = ROBOCASA_CORP_CONFIG["state_dim"]
    B = 2
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = adapted.to(device).to(dtype)
    model.train()
    images = {v: torch.randn(B, 3, 224, 224, device=device, dtype=dtype) for v in views}
    state = torch.randn(B, S, device=device, dtype=dtype)
    instr = ["pick up the mug", "open the cabinet drawer"]
    action = torch.randn(B, H, A, device=device, dtype=dtype)
    rep = model.extract(images, state, instr)
    loss = model.flow_matching_loss(images, state, instr, action)
    loss.backward()
    model.eval()
    with torch.no_grad():
        sampled = model.sample_action(images, state, instr)
    report["forward"] = {
        "device": device,
        "dtype": str(dtype),
        "tokens": list(rep.tokens.shape),
        "flow_loss": float(loss.detach()),
        "sample_shape": list(sampled.shape),
        "finite_loss": bool(torch.isfinite(loss)),
    }
    print(f"[5] GPU forward ok: tokens={tuple(rep.tokens.shape)} "
          f"loss={float(loss):.4f} sample={tuple(sampled.shape)}")

    (OUT / "smoke_real.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("REAL SMOKE PASSED ->", OUT / "smoke_real.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
