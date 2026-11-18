"""Estimate the Point of No Return by rollback branching from saved states.

At selected steps of a frozen rollout, save the MuJoCo state, restore it in a
fresh environment, and run N branches of the frozen policy. If success drops to
zero, that step is a candidate Point of No Return.
"""

from __future__ import annotations
import argparse, copy, json, sys
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


def branch(snap, first_obs, first_action, rng_state, replay_actions, env, policy, language, seed,
           max_steps, executed_k, exact_rng=False, replay_first_action=False, replay_all=False):
    if exact_rng and rng_state is not None:
        torch.set_rng_state(rng_state["torch"])
        np.random.set_state(rng_state["numpy"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    else:
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    # Restore in the same simulator to preserve controller and renderer state.
    restore(env, snap)
    env.env.sim.forward()
    obs = first_obs if first_obs is not None else env.env._get_observations()
    success = False; done = False; steps = 0
    restored = snapshot(env)
    restore_diff = 0.0
    for key in ("qpos", "qvel", "ctrl", "time", "mocap_pos", "mocap_quat", "userdata"):
        a = np.asarray(restored.get(key, []), dtype=float); b = np.asarray(snap.get(key, []), dtype=float)
        if a.size and b.size and a.shape == b.shape:
            restore_diff = max(restore_diff, float(np.max(np.abs(a - b))))
    action_diff = None
    replayed = False
    with torch.no_grad():
        while not success and not done and steps < max_steps:
            if replay_all and replay_actions is not None and steps < len(replay_actions):
                controller = np.asarray(replay_actions[steps:steps + executed_k], dtype=np.float32)
                if len(controller) == 0:
                    break
            elif replay_first_action and not replayed and first_action is not None:
                controller = np.asarray(first_action, dtype=np.float32)
                replayed = True
            else:
                po = policy_observation(obs, language, policy)
                with torch.autocast(device_type=torch.device("cuda").type, dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_available()):
                    controller = policy.predict(po)[0].float().cpu().numpy()
            if action_diff is None and first_action is not None:
                action_diff = float(np.max(np.abs(controller - first_action)))
            for k in range(min(executed_k, len(controller))):
                clipped = np.clip(controller[k], -1.0, 1.0)
                obs, reward, done, info = env.step(clipped); steps += 1
                success = is_success(env, info, reward, done)
                if success or done or steps >= max_steps: break
    return bool(success), steps, action_diff, restore_diff


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
    ap.add_argument("--replay-first-action", action="store_true")
    ap.add_argument("--replay-all-actions", action="store_true")
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
            snaps = {}; executed_actions = []; steps = 0; success = False; done = False
            if 0 in steps_wanted:
                snaps[0] = {"state": snapshot(env), "obs": copy.deepcopy(obs), "action": None,
                            "rng": {"torch": torch.get_rng_state(),
                                    "numpy": np.random.get_state(),
                                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
            with torch.no_grad():
                while not success and not done and steps < args.max_steps:
                    po = policy_observation(obs, language, policy)
                    with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                        enabled=args.device.startswith("cuda")):
                        controller = policy.predict(po)[0].float().cpu().numpy()
                    if steps == 0 and 0 in snaps and snaps[0]["action"] is None:
                        snaps[0]["action"] = controller.copy()
                    for k in range(args.executed_k):
                        act = np.clip(controller[k], -1.0, 1.0)
                        executed_actions.append(act.copy())
                        obs, reward, done, info = env.step(act); steps += 1
                        if steps in steps_wanted:
                            snaps[steps] = {"state": snapshot(env), "obs": copy.deepcopy(obs),
                                            "action": controller[k + 1:].copy(),
                                            "rng": {"torch": torch.get_rng_state(),
                                                    "numpy": np.random.get_state(),
                                                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
                        success = is_success(env, info, reward, done)
                        if success or done or steps >= args.max_steps: break
            for snap_step, pack in sorted(snaps.items()):
                snap = pack["state"]; first_obs = pack["obs"]; first_action = pack.get("action")
                rng_state = pack.get("rng")
                probs = []; restore_diffs = []; branch_details = []
                for b in range(args.branches):
                    replay = executed_actions[snap_step:] if args.replay_all_actions else None
                    ok, n, adiff, rdiff = branch(snap, first_obs, first_action, rng_state, replay, env, policy, language,
                                          args.seed if b == 0 else args.seed + 1000 * b,
                                          args.max_steps - snap_step, args.executed_k, exact_rng=(b == 0),
                                          replay_first_action=args.replay_first_action,
                                          replay_all=args.replay_all_actions)
                    probs.append(ok); restore_diffs.append(rdiff)
                    branch_details.append({"branch": b, "seed": args.seed if b == 0 else args.seed + 1000 * b,
                                           "success": bool(ok), "steps": n,
                                           "first_action_diff": adiff, "restore_diff": rdiff})
                records.append({"state_id": state_id, "snapshot_step": snap_step,
                                "branch_successes": int(sum(probs)), "branches": len(probs),
                                "first_action_diff": branch_details[0]["first_action_diff"] if branch_details else None,
                                "restore_diff": max(restore_diffs) if restore_diffs else None,
                                "branch_details": branch_details,
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
