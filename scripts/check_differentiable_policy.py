"""Smoke-test the frozen differentiable CoRP path and adapter gradients."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from corp.representation.extractor import Representation  # noqa: E402
from target_only_action_validation import configure  # noqa: E402
from dacsc.action_adapter import ResidualVisualAdapter  # noqa: E402
from dacsc.differentiable_policy import (  # noqa: E402
    _encode_frozen, sample_from_z0, forward_with_adapter, zero_init_equivalence,
)
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--state-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    import corp.action_generator as ag_module
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == args.suite)
    split = load_split(args.split_json); validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == args.lighting_id)
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    env = OffScreenRenderEnv(bddl_file_name=variant["base_bddl"], camera_heights=256,
                             camera_widths=256, render_gpu_device_id=args.render_gpu)
    try:
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        env.seed(args.seed); env.reset(); obs = env.set_init_state(states[args.state_id])
        for _ in range(SETTLING_STEPS): obs, _, _, _ = env.step(DUMMY_ACTION)
        po = policy_observation(obs, language, policy)
        torch.manual_seed(123)
        with torch.no_grad():
            z0 = torch.randn((1, policy.action_generator.action_horizon,
                              policy.action_generator.action_dim),
                             device=device, dtype=next(policy.parameters()).dtype)
        adapter = ResidualVisualAdapter(768, 8).to(device)
        zero = zero_init_equivalence(policy, po, adapter, z0)
        # Official path with the same z0 by patching torch.randn only for the
        # action generator module; restore immediately.
        orig_randn = ag_module.torch.randn
        ag_module.torch.randn = lambda *a, **k: z0
        try:
            with torch.no_grad():
                official = policy.predict(po)[0].float().cpu().numpy()
        finally:
            ag_module.torch.randn = orig_randn
        out = forward_with_adapter(policy, po, adapter, z0)
        ours = out["controller_actions"][0].detach().float().cpu().numpy()
        official_diff = float(np.max(np.abs(official - ours)))
        loss = out["normalized_actions"].square().mean()
        loss.backward()
        grads = [p.grad for p in adapter.parameters() if p.grad is not None]
        report = {"zero_init": zero, "official_vs_ours_max_abs": official_diff,
                  "adapter_grad_params": len(grads),
                  "adapter_grad_norm": float(sum(float(g.norm()) for g in grads)),
                  "z0_shape": list(z0.shape), "action_shape": list(ours.shape)}
        args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
