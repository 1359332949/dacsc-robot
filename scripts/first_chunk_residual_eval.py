"""Evaluate a source-paired first-chunk action residual on target dev."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from target_only_action_validation import configure, visual_from_obs  # noqa: E402
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--role", default="target_dev")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2014)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == args.suite)
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == args.lighting_id)
    split = load_split(args.split_json)
    validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    target_ids = resolve_indices(split, args.role, "")
    pairs = json.loads(args.pairs.read_text())["records"]
    src_state = np.stack([np.asarray(r["domains"]["shift"]["state8"]) for r in pairs])
    src_visual = np.stack([np.asarray(r["domains"]["shift"]["visual_mean"]) for r in pairs])
    src_delta = np.stack([
        np.asarray(r["domains"]["clean"]["actions"]) - np.asarray(r["domains"]["shift"]["actions"])
        for r in pairs
    ])
    src_state_mean, src_state_std = src_state.mean(0), src_state.std(0) + 1e-6
    src_visual_mean, src_visual_std = src_visual.mean(0), src_visual.std(0) + 1e-6
    src_state_n = (src_state - src_state_mean) / src_state_std
    src_visual_n = (src_visual - src_visual_mean) / src_visual_std
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    records = []
    for state_id in target_ids:
        env = OffScreenRenderEnv(bddl_file_name=variant["env_bddl_name"], camera_heights=256,
                                 camera_widths=256, render_gpu_device_id=args.render_gpu)
        try:
            np.random.seed(args.seed); torch.manual_seed(args.seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
            env.seed(args.seed); env.reset(); obs = env.set_init_state(states[state_id])
            for _ in range(SETTLING_STEPS):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            steps = 0; success = False; done = False; chunk_index = 0
            while not success and not done and steps < args.max_steps:
                po = policy_observation(obs, language, policy)
                if chunk_index == 0:
                    state8 = po["state"][0].detach().cpu().numpy().astype(float)
                    visual, _ = visual_from_obs(policy, po)
                    visual = visual.mean(dim=1)[0].detach().float().cpu().numpy()
                    q = np.concatenate([(state8 - src_state_mean) / src_state_std,
                                        (visual - src_visual_mean) / src_visual_std])
                    qs = np.concatenate([src_state_n, src_visual_n], axis=1)
                    dist = np.linalg.norm(qs - q, axis=1)
                    nn = np.argsort(dist)[:args.k]
                    delta = src_delta[nn].mean(0)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                        controller = policy.predict(po)[0].float().cpu().numpy()
                    controller = controller + args.alpha * delta
                    controller = np.clip(controller, -1.0, 1.0)
                    # Use the same controller for all 12 actions in chunk 0.
                    actions = controller
                else:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=device.type == "cuda"):
                        actions = policy.predict(po)[0].float().cpu().numpy()
                for k in range(args.executed_k):
                    raw = actions[k]
                    clipped = np.clip(raw, -1.0, 1.0)
                    obs, reward, done, info = env.step(clipped)
                    steps += 1
                    success = is_success(env, info, reward, done)
                    if success or done or steps >= args.max_steps: break
                chunk_index += 1
            records.append({"state_id": state_id, "seed": args.seed, "success": bool(success),
                            "steps": int(steps), "k": args.k, "alpha": args.alpha})
            print(json.dumps(records[-1]), flush=True)
        finally:
            env.close()
    payload = {"schema": "first_chunk_residual_eval_v1", "suite": args.suite,
               "lighting_id": args.lighting_id, "role": args.role, "k": args.k,
               "alpha": args.alpha, "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "success": sum(r["success"] for r in records), "n": len(records)}))


if __name__ == "__main__":
    main()
