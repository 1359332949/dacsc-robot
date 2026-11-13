"""V0: verify an identity VisualTokenGate is a true no-op on CoRP forward."""

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
    dim = 384
    gate = VisualTokenGate(base.view_names, dim, mode="identity").to(args.device)
    wrapped = GatedRepresentationExtractor(base, gate).eval()

    images = {name: torch.randn(args.batch, 3, 224, 224, device=args.device)
              for name in base.view_names}
    state = torch.randn(args.batch, policy.config["state_dim"], device=args.device)
    instruction = ["pick up the black bowl"] * args.batch

    with torch.no_grad():
        rep0 = base(images=images, state=state, instruction=instruction)
        rep1 = wrapped(images=images, state=state, instruction=instruction)
        tokens_equal = torch.equal(rep0.tokens, rep1.tokens)
        pooled_equal = torch.equal(rep0.pooled_language, rep1.pooled_language)
        dtype_equal = rep0.action_dtype == rep1.action_dtype
        # Same action latent seed on both representations.
        def sample(rep):
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
                torch.manual_seed(1234)
                return policy.action_generator.sample(rep).detach()
        a0, a1 = sample(rep0), sample(rep1)
        action_equal = torch.equal(a0, a1)
        # Non-identity gate should change tokens/actions (mechanism diagnostic only).
        gate.set_uniform(0.5); gate.mode = "scale"
        rep2 = wrapped(images=images, state=state, instruction=instruction)
        tokens_changed = not torch.equal(rep2.tokens, rep0.tokens)
        a2 = sample(rep2)
        action_changed = not torch.equal(a2, a0)
    report = {
        "device": str(args.device), "view_names": list(base.view_names),
        "identity_tokens_equal": bool(tokens_equal),
        "identity_pooled_equal": bool(pooled_equal),
        "identity_dtype_equal": bool(dtype_equal),
        "identity_action_equal": bool(action_equal),
        "scale_tokens_changed": bool(tokens_changed),
        "scale_action_changed": bool(action_changed),
        "max_token_abs_diff": float((rep0.tokens - rep1.tokens).abs().max()) if not tokens_equal else 0.0,
        "max_action_abs_diff": float((a0 - a1).abs().max()) if not action_equal else 0.0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not (tokens_equal and pooled_equal and dtype_equal and action_equal):
        raise SystemExit("V0 identity check failed")


if __name__ == "__main__":
    main()
