"""Evaluate a proprioceptive grasp-reflex around a frozen VLA policy.

No target labels and no simulator object truth are used at test time. The
reflex triggers when the gripper has closed but eef z has not risen within a
short window; it opens the gripper, retreats, and replans.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from target_only_action_validation import configure  # noqa: E402
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa: E402


def run(env, policy, language, state, seed, args, mode):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    env.seed(seed); env.reset(); obs = env.set_init_state(state)
    for _ in range(SETTLING_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    steps = 0; success = False; done = False; interventions = 0
    closed_at = None; z_at_close = None; z_after_close = []
    while not success and not done and steps < args.max_steps:
        po = policy_observation(obs, language, policy)
        with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                            enabled=args.device.startswith("cuda")):
            controller = policy.predict(po)[0].float().cpu().numpy()
        for k in range(args.executed_k):
            raw = controller[k]; clipped = np.clip(raw, -1.0, 1.0)
            obs, reward, done, info = env.step(clipped)
            steps += 1
            success = is_success(env, info, reward, done)
            st = policy_observation(obs, language, policy)["state"][0].detach().cpu().numpy().astype(float)
            gap = abs(st[6]) + abs(st[7])
            if gap < args.gripper_closed_threshold:
                if closed_at is None:
                    closed_at = steps; z_at_close = st[2]; z_after_close = []
                else:
                    z_after_close.append(st[2])
            if success or done or steps >= args.max_steps: break
            if mode in ("reflex", "interrupt") and interventions < args.max_interventions:
                if closed_at is not None and len(z_after_close) >= args.lift_window:
                    lift = max(z_after_close) - z_at_close
                    if lift < args.lift_threshold:
                        interventions += 1
                        if mode == "reflex":
                            # Open gripper, lift/retreat modestly, then force a replan.
                            reflex = np.array([0.0, 0.0, args.retreat_z, 0.0, 0.0, 0.0, -1.0], dtype=float)
                            for _ in range(args.reflex_steps):
                                obs, reward, done, info = env.step(reflex)
                                steps += 1
                                success = is_success(env, info, reward, done)
                                if success or done or steps >= args.max_steps: break
                        closed_at = None; z_at_close = None; z_after_close = []
                        break
                if closed_at is None:
                    z_after_close = []
    return {"success": bool(success), "steps": int(steps), "interventions": interventions,
            "hit_max_steps": bool(steps >= args.max_steps and not success)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--role", default="target_dev")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--seeds", default="7777")
    ap.add_argument("--modes", default="frozen,reflex")
    ap.add_argument("--state-ids", default="")
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--gripper-closed-threshold", type=float, default=0.02)
    ap.add_argument("--lift-window", type=int, default=8)
    ap.add_argument("--lift-threshold", type=float, default=0.01)
    ap.add_argument("--retreat-z", type=float, default=0.35)
    ap.add_argument("--reflex-steps", type=int, default=8)
    ap.add_argument("--max-interventions", type=int, default=2)
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
    split = load_split(args.split_json)
    validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == args.lighting_id)
    ids = resolve_indices(split, args.role, "")
    if args.state_ids.strip():
        req = [int(x) for x in args.state_ids.split(",") if x.strip()]
        outside = set(req) - set(ids)
        if outside: raise ValueError(f"outside role: {sorted(outside)}")
        ids = req
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    policy = load_policy(args.checkpoint, device=torch.device(args.device)); policy.eval()
    records = []
    for state_id in ids:
        for seed in [int(x) for x in args.seeds.split(",")]:
            for mode in args.modes.split(","):
                env = OffScreenRenderEnv(bddl_file_name=variant["env_bddl_name"], camera_heights=256,
                                         camera_widths=256, render_gpu_device_id=args.render_gpu)
                try:
                    r = run(env, policy, language, states[state_id], seed, args, mode)
                    r.update({"state_id": state_id, "seed": seed, "mode": mode})
                    records.append(r); print(json.dumps(r), flush=True)
                finally:
                    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "reflex_recovery_eval_v1", "records": records}, indent=2) + "\n")
    for mode in args.modes.split(","):
        rs = [r for r in records if r["mode"] == mode]
        print(mode, sum(r["success"] for r in rs), "/", len(rs))


if __name__ == "__main__":
    main()
