"""Estimate the Point of No Return by rollback branching from saved states.

At selected steps of a frozen rollout, save the MuJoCo state, restore it in a
fresh environment, and run N branches of the frozen policy. If success drops to
zero, that step is a candidate Point of No Return.
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
from dacsc.state_pairing import snapshot, restore, model_compatible  # noqa: E402
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa: E402


def branch(snap, env, policy, language, seed, max_steps, executed_k):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    env.seed(seed); env.reset()
    compat = model_compatible(snapshot(env), snap)
    if not all(compat.values()):
        raise ValueError(f"rollback model incompatible: {compat}")
    restore(env, snap)
    env.env.sim.forward()
    obs = env.env._get_observations()
    success = False; done = False; steps = 0
    with torch.no_grad():
        while not success and not done and steps < max_steps:
            po = policy_observation(obs, language, policy)
            with torch.autocast(device_type=torch.device("cuda").type, dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                controller = policy.predict(po)[0].float().cpu().numpy()
            for k in range(executed_k):
                clipped = np.clip(controller[k], -1.0, 1.0)
                obs, reward, done, info = env.step(clipped); steps += 1
                success = is_success(env, info, reward, done)
                if success or done or steps >= max_steps: break
    return bool(success), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--state-ids", default="")
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--snapshot-steps", default="60,80,100,120,140")
    ap.add_argument("--branches", type=int, default=3)
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
    split = load_split(args.split_json); validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == args.lighting_id)
    ids = resolve_indices(split, "target_dev", args.state_ids)
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    policy = load_policy(args.checkpoint, device=torch.device(args.device)); policy.eval()
    steps_wanted = [int(x) for x in args.snapshot_steps.split(",")]
    records = []
    for state_id in ids:
        env = OffScreenRenderEnv(bddl_file_name=variant["env_bddl_name"], camera_heights=256,
                                 camera_widths=256, render_gpu_device_id=args.render_gpu)
        try:
            np.random.seed(args.seed); torch.manual_seed(args.seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
            env.seed(args.seed); env.reset(); obs = env.set_init_state(states[state_id])
            for _ in range(SETTLING_STEPS): obs, _, _, _ = env.step(DUMMY_ACTION)
            snaps = {}; steps = 0; success = False; done = False
            with torch.no_grad():
                while not success and not done and steps < args.max_steps:
                    po = policy_observation(obs, language, policy)
                    with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                        enabled=args.device.startswith("cuda")):
                        controller = policy.predict(po)[0].float().cpu().numpy()
                    for k in range(args.executed_k):
                        obs, reward, done, info = env.step(np.clip(controller[k], -1.0, 1.0)); steps += 1
                        if steps in steps_wanted:
                            snaps[steps] = snapshot(env)
                        success = is_success(env, info, reward, done)
                        if success or done or steps >= args.max_steps: break
            for snap_step, snap in sorted(snaps.items()):
                probs = []
                for b in range(args.branches):
                    benv = OffScreenRenderEnv(bddl_file_name=variant["env_bddl_name"], camera_heights=256,
                                              camera_widths=256, render_gpu_device_id=args.render_gpu)
                    try:
                        ok, n = branch(snap, benv, policy, language, args.seed + 1000 * b,
                                       args.max_steps - snap_step, args.executed_k)
                        probs.append(ok)
                    finally:
                        benv.close()
                records.append({"state_id": state_id, "snapshot_step": snap_step,
                                "branch_successes": int(sum(probs)), "branches": len(probs),
                                "final_success": bool(success), "final_steps": steps})
        finally:
            env.close()
    by_state = {}
    for r in records:
        by_state.setdefault(str(r["state_id"]), []).append(r)
    summary = {}
    for st, rows in by_state.items():
        rows = sorted(rows, key=lambda x: x["snapshot_step"])
        p = [r["branch_successes"] / max(r["branches"], 1) for r in rows]
        ponr = None
        for i in range(len(rows)):
            if p[i] == 0 and any(x > 0 for x in p[:i]):
                ponr = rows[i]["snapshot_step"]; break
        summary[st] = {"branch_success_rates": p, "snapshot_steps": [r["snapshot_step"] for r in rows],
                       "ponr_step": ponr, "final_success": rows[0]["final_success"] if rows else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"records": records, "summary": summary}, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "n": len(records)}))


if __name__ == "__main__":
    main()
