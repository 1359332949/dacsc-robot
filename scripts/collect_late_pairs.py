"""Collect late-window (pre-grasp/grasp-closure) Clean/Shift pairs."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from target_only_action_validation import configure  # noqa: E402
from dacsc.state_pairing import snapshot, restore, model_compatible, compare, camera_compare, digest, digest_model, digest_controller  # noqa: E402
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa: E402
from scripts.collect_action_consistency_pairs import encode_actions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--role", default="source_fit")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--state-ids", default="")
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--warmup-chunks", type=int, default=6)
    ap.add_argument("--probe-chunks", type=int, default=4)
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    import os
    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise RuntimeError("requires OSMesa")
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == args.suite)
    split = load_split(args.split_json); validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == args.lighting_id)
    ids = resolve_indices(split, args.role, args.state_ids)
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    records = []
    for state_id in ids:
        shift = OffScreenRenderEnv(bddl_file_name=variant["env_bddl_name"], camera_heights=256,
                                   camera_widths=256, render_gpu_device_id=args.render_gpu)
        clean = OffScreenRenderEnv(bddl_file_name=variant["base_bddl"], camera_heights=256,
                                   camera_widths=256, render_gpu_device_id=args.render_gpu)
        try:
            np.random.seed(args.seed); torch.manual_seed(args.seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
            shift.seed(args.seed); shift.reset(); obs = shift.set_init_state(states[state_id])
            for _ in range(SETTLING_STEPS): obs, _, _, _ = shift.step(DUMMY_ACTION)
            steps = 0; chunk = 0; success = False; done = False
            while not success and not done and steps < args.max_steps:
                po_s = policy_observation(obs, language, policy)
                torch.manual_seed(args.seed + 100003 * state_id + chunk)
                z0 = torch.randn((1, policy.action_generator.action_horizon, policy.action_generator.action_dim),
                                 device=device, dtype=next(policy.parameters()).dtype)
                tok_s, norm_s, ctrl_s, lang_s, state_s_tok, pooled_s = encode_actions(policy, po_s, z0)
                if chunk >= args.warmup_chunks and chunk < args.warmup_chunks + args.probe_chunks:
                    snap = snapshot(shift)
                    clean.seed(args.seed + 991 * chunk); clean.reset()
                    clean.env.sim.model.body_pos[:] = np.asarray(snap["model_signature"]["body_pos"], float)
                    clean.env.sim.model.body_quat[:] = np.asarray(snap["model_signature"]["body_quat"], float)
                    clean.env.sim.forward()
                    compat = model_compatible(snapshot(clean), snap, strict_bodies=True, strict_geometry=True, strict_physics=True)
                    restore(clean, snap); clean._update_observables(force=True)
                    obs_c = clean.env._get_observations()
                    snap_c = snapshot(clean)
                    state_check = compare(snap, snap_c, tol={"body_xpos": 1e-3, "eef_pos": 1e-3, "body_xquat_rad": 2e-3})
                    cam_check = camera_compare(snap, snap_c, expect_equal=True, tol=1e-3)
                    po_c = policy_observation(obs_c, language, policy)
                    state8 = float(np.max(np.abs(po_s["state"][0].detach().cpu().numpy() - po_c["state"][0].detach().cpu().numpy())))
                    tok_c, norm_c, ctrl_c, lang_c, state_c_tok, pooled_c = encode_actions(policy, po_c, z0)
                    gc = ctrl_c[0, :, 6].float().cpu().numpy(); gs = ctrl_s[0, :, 6].float().cpu().numpy()
                    records.append({
                        "state_id": int(state_id), "role": args.role, "seed": args.seed,
                        "chunk_index": int(chunk), "step": int(steps),
                        "pairing_pass": bool(compat["pass"] and state_check["all_pass"] and cam_check["pass"] and state8 < 1e-3),
                        "model_compatibility_pass": bool(compat["pass"]),
                        "physical_state_compare_pass": bool(state_check["all_pass"]),
                        "camera_compare_pass": bool(cam_check["pass"]),
                        "state8_max_abs": state8,
                        "shift_snapshot_digest": digest(snap), "clean_snapshot_digest": digest(snap_c),
                        "shift_model_digest": digest_model(snap), "clean_model_digest": digest_model(snap_c),
                        "shift_controller_digest": digest_controller(snap), "clean_controller_digest": digest_controller(snap_c),
                        "z0_sha256": hashlib.sha256(z0.detach().float().cpu().numpy().tobytes()).hexdigest(),
                        "z0": z0[0].float().cpu().numpy().tolist(),
                        "shift_visual_tokens": tok_s[0].float().cpu().numpy().tolist(),
                        "clean_visual_tokens": tok_c[0].float().cpu().numpy().tolist(),
                        "shift_normalized_actions": norm_s[0].float().cpu().numpy().tolist(),
                        "clean_normalized_actions": norm_c[0].float().cpu().numpy().tolist(),
                        "shift_controller_actions": ctrl_s[0].float().cpu().numpy().tolist(),
                        "clean_controller_actions": ctrl_c[0].float().cpu().numpy().tolist(),
                        "language_tokens": lang_s[0].float().cpu().numpy().tolist(),
                        "state_token": state_s_tok[0].float().cpu().numpy().tolist(),
                        "shift_state_token": state_s_tok[0].float().cpu().numpy().tolist(),
                        "clean_state_token": state_c_tok[0].float().cpu().numpy().tolist(),
                        "pooled_language": pooled_s[0].float().cpu().numpy().tolist(),
                        "gripper_max_abs_diff": float(np.max(np.abs(gc - gs))),
                        "gripper_mean_abs_diff": float(np.mean(np.abs(gc - gs))),
                    })
                for k in range(min(args.executed_k, ctrl_s.shape[1])):
                    action = np.clip(ctrl_s[0, k].cpu().numpy(), -1.0, 1.0)
                    obs, reward, done, info = shift.step(action); steps += 1
                    success = is_success(shift, info, reward, done)
                    if success or done or steps >= args.max_steps: break
                chunk += 1
            print(json.dumps({"state_id": int(state_id), "chunks": chunk, "success": bool(success), "steps": steps, "records": len(records)}), flush=True)
        finally:
            shift.close(); clean.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "late_pairs_v1", "role": args.role, "lighting_id": args.lighting_id,
                                    "warmup_chunks": args.warmup_chunks, "probe_chunks": args.probe_chunks,
                                    "records": records}, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "n": len(records)}))


if __name__ == "__main__":
    main()
