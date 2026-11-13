"""V0: verify identity VisualTokenGate is a true no-op on the CoRP forward path."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP"))
from corp import load_policy  # noqa: E402
from dacsc.visual_gate import GatedRepresentationExtractor, VisualTokenGate  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("/tmp/v0_identity_gate.json"))
    args = ap.parse_args()
    policy = load_policy(CKPT, device=args.device)
    policy.eval()
    base = policy.representation_extractor
    images = {name: torch.randn(args.batch, 3, 224, 224, device=args.device)
              for name in base.view_names}
    state = torch.randn(args.batch, policy.config["state_dim"], device=args.device)
    instruction = ["pick up the black bowl"] * args.batch

    raw = {}
    def capture(_m, _inp, out):
        raw["last"] = out.detach().clone()
    hook = base.visual_encoder.register_forward_hook(capture)
    with torch.no_grad():
        rep0 = base(images=images, state=state, instruction=instruction)
        raw0 = raw["last"]
        dim = int(raw0.shape[-1])
        gate = VisualTokenGate(base.view_names, dim, mode="identity").to(args.device)
        wrapped = GatedRepresentationExtractor(base, gate).eval()
        rep1 = wrapped(images=images, state=state, instruction=instruction)
        raw1 = raw["last"]
        tokens_equal = torch.equal(rep0.tokens, rep1.tokens)
        raw_equal = torch.equal(raw0, raw1)
        pooled_equal = torch.equal(rep0.pooled_language, rep1.pooled_language)
        dtype_equal = rep0.action_dtype == rep1.action_dtype

        def sample(rep):
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
                torch.manual_seed(1234)
                return policy.action_generator.sample(rep).detach()
        a0, a1 = sample(rep0), sample(rep1)
        action_equal = torch.equal(a0, a1)
        gate.mode = "suppression"; gate.set_uniform(0.5)
        rep2 = wrapped(images=images, state=state, instruction=instruction)
        tokens_changed = not torch.equal(rep2.tokens, rep0.tokens)
        a2 = sample(rep2)
        action_changed = not torch.equal(a2, a0)
    hook.remove()
    report = {
        "device": str(args.device), "view_names": list(base.view_names), "patch_dim": dim,
        "identity_raw_patches_equal": bool(raw_equal),
        "identity_representation_equal": bool(tokens_equal),
        "identity_pooled_equal": bool(pooled_equal),
        "identity_dtype_equal": bool(dtype_equal),
        "identity_action_equal": bool(action_equal),
        "suppression_tokens_changed": bool(tokens_changed),
        "suppression_action_changed": bool(action_changed),
        "max_representation_abs_diff": float((rep0.tokens - rep1.tokens).abs().max()) if not tokens_equal else 0.0,
        "max_action_abs_diff": float((a0 - a1).abs().max()) if not action_equal else 0.0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    required = (raw_equal and tokens_equal and pooled_equal and dtype_equal and action_equal and tokens_changed)
    if not required:
        raise SystemExit("V0 identity/suppression interface check failed")


if __name__ == "__main__":
    main()
