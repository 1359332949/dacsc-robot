"""Diagnose adapter residual magnitude and action change per state/z0."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from target_only_action_validation import configure  # noqa: E402
from dacsc.action_adapter import ResidualVisualAdapter  # noqa: E402
from dacsc.differentiable_policy import _encode_frozen, forward_with_adapter  # noqa: E402
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--domain", choices=["shift", "clean"], default="shift")
    ap.add_argument("--state-ids", required=True)
    ap.add_argument("--env-seed", type=int, default=7777)
    ap.add_argument("--z0-seeds", default="7777,8888,9999")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    import os
    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise RuntimeError("adapter support analysis requires MUJOCO_GL=osmesa")
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == args.suite)
    split = load_split(args.split_json); validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == args.lighting_id)
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    ids = resolve_indices(split, "target_dev", args.state_ids)
    bddl = variant["env_bddl_name"] if args.domain == "shift" else variant["base_bddl"]
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    ck = torch.load(args.adapter, map_location="cpu")
    adapter = ResidualVisualAdapter(768, ck["rank"]).to(device); adapter.load_state_dict(ck["adapter"]); adapter.eval()
    zero = ResidualVisualAdapter(768, ck["rank"]).to(device); zero.eval()
    zseeds = [int(x) for x in args.z0_seeds.split(",")]
    rows = []
    state_actions = {}
    for state_id in ids:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                 render_gpu_device_id=args.render_gpu)
        try:
            np.random.seed(args.env_seed); torch.manual_seed(args.env_seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.env_seed)
            env.seed(args.env_seed); env.reset(); obs = env.set_init_state(states[state_id])
            for _ in range(SETTLING_STEPS): obs, _, _, _ = env.step(DUMMY_ACTION)
            po = policy_observation(obs, language, policy)
            pieces = _encode_frozen(policy, po)
            h = pieces["visual_tokens"]
            with torch.no_grad(): a = adapter(h)
            res = a - h
            per_view = []
            for v in range(2):
                hv = h[:, v * 48:(v + 1) * 48]
                rv = res[:, v * 48:(v + 1) * 48]
                per_view.append(float(rv.norm(dim=-1).mean() / (hv.norm(dim=-1).mean() + 1e-6)))
            for zs in zseeds:
                torch.manual_seed(zs)
                z0 = torch.randn((1, policy.action_generator.action_horizon,
                                  policy.action_generator.action_dim), device=device, dtype=torch.float32)
                af = forward_with_adapter(policy, po, zero, z0)["controller_actions"][0].detach().float().cpu().numpy()
                aa = forward_with_adapter(policy, po, adapter, z0)["controller_actions"][0].detach().float().cpu().numpy()
                d = aa - af
                state_actions.setdefault(int(state_id), {"f": [], "a": []})
                state_actions[int(state_id)]["f"].append(af)
                state_actions[int(state_id)]["a"].append(aa)
                rows.append({
                    "state_id": int(state_id), "env_seed": args.env_seed, "z0_seed": zs,
                    "residual_ratio_mean": float(per_view[0] + per_view[1]) / 2,
                    "residual_ratio_views": per_view,
                    "action_delta_rmse": float(np.sqrt(np.mean(d ** 2))),
                    "action_delta_xyz": float(np.sqrt(np.mean(d[:, :3] ** 2))),
                    "action_delta_rot": float(np.sqrt(np.mean(d[:, 3:6] ** 2))),
                    "action_delta_grip": float(np.sqrt(np.mean(d[:, 6] ** 2))),
                })
        finally:
            env.close()
    for r in rows:
        acts = state_actions[int(r["state_id"])]
        f = np.stack(acts["f"]); a = np.stack(acts["a"])
        r["frozen_z0_dispersion"] = float(np.sqrt(np.mean(np.var(f, axis=0))))
        r["adapter_z0_dispersion"] = float(np.sqrt(np.mean(np.var(a, axis=0))))
        r["dispersion_ratio"] = float(r["adapter_z0_dispersion"] / (r["frozen_z0_dispersion"] + 1e-9))
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "n": len(rows)}))


if __name__ == "__main__":
    main()
