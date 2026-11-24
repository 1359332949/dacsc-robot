"""Frozen Clean/Shift action and failure-phase diagnostic.

No adaptation: this script only records what the frozen policy does on the same
state/seed in Clean and Shift, plus a coarse phase proxy from the 8-D state.
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
from dacsc.tta_protocol import load_split, resolve_indices, role_ids, validate_split_identity  # noqa: E402


def classify_phase(state8, initial_z, max_gain):
    # state8: eef_pos(3), rotvec(3), gripper(2)
    z_gain = float(state8[2] - initial_z)
    gripper = float(np.min(state8[6:8]))
    if max_gain < 0.05:
        return "no_lift"
    if gripper > 0.02:
        return "grasp_or_hold_not_closed"
    if z_gain < 0.05:
        return "lift_failed"
    return "transport_or_place_timeout"


def run_episode(env, policy, language, initial_state, seed, max_steps, executed_k):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    env.seed(seed); env.reset(); obs = env.set_init_state(initial_state)
    for _ in range(SETTLING_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    po = policy_observation(obs, language, policy)
    initial_z = float(po["state"][0, 2])
    actions, states, chunks = [], [], []
    success = False; steps = 0; done = False
    with torch.no_grad():
        while not success and not done and steps < max_steps:
            po = policy_observation(obs, language, policy)
            state8 = po["state"][0].detach().cpu().numpy().astype(float)
            states.append(state8)
            controller = policy.predict(po)[0].float().cpu().numpy()
            chunk_actions = []
            for k in range(executed_k):
                raw = controller[k]
                clipped = np.clip(raw, -1.0, 1.0)
                actions.append(clipped.astype(float)); chunk_actions.append(clipped.astype(float))
                obs, reward, done, info = env.step(clipped)
                steps += 1
                success = is_success(env, info, reward, done)
                if success or done or steps >= max_steps:
                    break
            chunks.append(np.stack(chunk_actions))
    final_state = policy_observation(obs, language, policy)["state"][0].detach().cpu().numpy().astype(float)
    zs = [s[2] for s in states] + [final_state[2]]
    max_gain = float(max(zs) - initial_z)
    phase = "success" if success else classify_phase(final_state, initial_z, max_gain)
    arr = np.stack(actions) if actions else np.zeros((0, 7))
    return {
        "seed": int(seed), "success": bool(success), "steps": int(steps),
        "failure_phase": phase, "max_eef_z_gain": max_gain,
        "initial_z": initial_z, "final_state8": final_state.tolist(),
        "action_abs_mean": float(np.abs(arr).mean()) if len(arr) else 0.0,
        "action_norm_mean": float(np.linalg.norm(arr, axis=1).mean()) if len(arr) else 0.0,
        "action_saturation_fraction": float(np.any(np.abs(arr) > 1.0, axis=1).mean()) if len(arr) else 0.0,
        "chunk_action_means": [np.asarray(c).mean(axis=0).tolist() for c in chunks],
        "chunk_action_norms": [float(np.linalg.norm(c, axis=1).mean()) if len(c) else 0.0 for c in chunks],
        "actions": arr.tolist(),
        "states8": [s.tolist() for s in states],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--role", default="target_dev")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="2014")
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
    ids = resolve_indices(split, args.role, "")
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    base_bddl = variant["base_bddl"]; shift_bddl = variant["env_bddl_name"]
    language = BDDLUtils.get_problem_info(base_bddl)["language_instruction"]
    seeds = [int(x) for x in args.seeds.split(",")]
    records = []
    for state_id in ids:
        for seed in seeds:
            for domain, bddl in (("clean", base_bddl), ("shift", shift_bddl)):
                env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                                         camera_widths=256, render_gpu_device_id=args.render_gpu)
                try:
                    rec = run_episode(env, policy, language, states[state_id], seed,
                                      args.max_steps, args.executed_k)
                    rec.update({"state_id": state_id, "domain": domain, "split_role": args.role})
                    records.append(rec); print(json.dumps({k: rec[k] for k in rec if k not in ("actions", "states8", "chunk_action_means")}), flush=True)
                finally:
                    env.close()
    payload = {"schema": "action_diagnostic_v1", "suite": args.suite, "lighting_id": args.lighting_id,
               "role": args.role, "ids": ids, "seeds": seeds, "records": records,
               "source_file_sha256": __import__("dacsc.tta_protocol", fromlist=["sha256_file"]).sha256_file(task["state_split"]["source_file"])}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "n": len(records)}))


if __name__ == "__main__":
    main()
