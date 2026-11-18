"""Collect same-physical-state Clean/Shift action-consistency pairs.

For each shift replan boundary:
1. save the full MuJoCo snapshot;
2. encode the shift observation and sample an action chunk with explicit z0;
3. restore the same snapshot in a Clean twin environment;
4. encode the Clean twin and sample with the SAME z0;
5. check state pairing/model compatibility and save tokens + actions.
"""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from corp.representation.extractor import Representation  # noqa: E402
from target_only_action_validation import configure  # noqa: E402
from dacsc.action_adapter import ResidualVisualAdapter  # noqa: E402
from dacsc.differentiable_policy import _encode_frozen, sample_from_z0  # noqa: E402
from dacsc.state_pairing import (  # noqa: E402
    snapshot, restore, model_compatible, compare, camera_compare,
    digest, digest_model, digest_controller,
)
from dacsc.tta_protocol import load_split, resolve_indices, validate_split_identity  # noqa: E402


def encode_actions(policy, observation, z0):
    pieces = _encode_frozen(policy, observation)
    visual_language = torch.cat(
        [pieces["visual_tokens"], pieces["language_tokens"].to(pieces["visual_tokens"].dtype)], dim=1
    )
    rep = Representation(
        tokens=torch.cat([visual_language, pieces["state_token"]], dim=1),
        pooled_language=pieces["pooled_language"],
        action_dtype=visual_language.dtype,
    )
    with torch.no_grad():
        normalized = sample_from_z0(policy, rep, z0)
        controller = policy.normalizer.unnormalize_action(normalized)
    return (pieces["visual_tokens"], normalized, controller,
            pieces["language_tokens"], pieces["state_token"], pieces["pooled_language"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--role", default="source_fit")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--state-ids", default="")
    ap.add_argument("--max-chunks", type=int, default=2)
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    import os
    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise RuntimeError("pair collection requires MUJOCO_GL=osmesa for deterministic rendering")
    if args.out.exists(): raise FileExistsError(args.out)
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
    adapter = ResidualVisualAdapter(768, 8).to(device); adapter.eval()
    for p in adapter.parameters(): p.requires_grad_(False)
    records = []
    for state_id in ids:
        shift = OffScreenRenderEnv(bddl_file_name=variant["env_bddl_name"], camera_heights=256,
                                   camera_widths=256, render_gpu_device_id=args.render_gpu)
        clean = OffScreenRenderEnv(bddl_file_name=variant["base_bddl"], camera_heights=256,
                                   camera_widths=256, render_gpu_device_id=args.render_gpu)
        try:
            np.random.seed(args.seed); torch.manual_seed(args.seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
            shift.seed(args.seed); shift.reset(); obs_s = shift.set_init_state(states[state_id])
            for _ in range(SETTLING_STEPS): obs_s, _, _, _ = shift.step(DUMMY_ACTION)
            steps = 0; chunk = 0; success = False; done = False
            while not success and not done and steps < args.max_steps and chunk < args.max_chunks:
                snap = snapshot(shift)
                obs_clean = None
                clean.seed(args.seed + 991 * chunk); clean.reset()
                # Robot base/body poses are randomized by env.reset() and are
                # not part of MuJoCo qpos; copy them from the shift snapshot.
                clean.env.sim.model.body_pos[:] = np.asarray(snap["model_signature"]["body_pos"], dtype=float)
                clean.env.sim.model.body_quat[:] = np.asarray(snap["model_signature"]["body_quat"], dtype=float)
                clean.env.sim.forward()
                snap_clean_probe = snapshot(clean)
                compat = model_compatible(snap_clean_probe, snap, strict_bodies=True,
                                          strict_geometry=True, strict_physics=True)
                if not compat["pass"]:
                    raise RuntimeError(f"model compatibility failed for state {state_id}: {compat}")
                restore(clean, snap)
                clean._update_observables(force=True)
                obs_clean = clean.env._get_observations()
                snap_clean = snapshot(clean)
                # qpos/qvel/act must be exact; derived body poses are allowed a
                # sub-millimetre tolerance because model body/joint metadata can
                # differ slightly across lighting BDDL variants.
                state_check = compare(snap, snap_clean, tol={
                    "body_xpos": 1e-3, "eef_pos": 1e-3,
                    "body_xquat_rad": 2e-3,
                })
                cam_check = camera_compare(snap, snap_clean, expect_equal=True, tol=1e-3)
                if not state_check["all_pass"]:
                    raise RuntimeError(f"physical state pairing failed for state {state_id}: {state_check}")
                if not cam_check["pass"]:
                    raise RuntimeError(f"camera pairing failed for state {state_id}: {cam_check}")
                po_s = policy_observation(obs_s, language, policy)
                po_c = policy_observation(obs_clean, language, policy)
                state_s = po_s["state"][0].detach().cpu().numpy().astype(float)
                state_c = po_c["state"][0].detach().cpu().numpy().astype(float)
                state8_max = float(np.max(np.abs(state_s - state_c)))
                torch.manual_seed(args.seed + 100003 * state_id + chunk)
                z0 = torch.randn((1, policy.action_generator.action_horizon, policy.action_generator.action_dim),
                                 device=device, dtype=next(policy.parameters()).dtype)
                zhash = hashlib.sha256(z0.detach().float().cpu().numpy().tobytes()).hexdigest()
                tok_s, norm_s, ctrl_s, lang_s, state_s_tok, pooled_s = encode_actions(policy, po_s, z0)
                tok_c, norm_c, ctrl_c, lang_c, state_c_tok, pooled_c = encode_actions(policy, po_c, z0)
                records.append({
                    "state_id": int(state_id), "seed": args.seed, "chunk_index": chunk,
                    "step": int(steps),
                    "pairing_pass": bool(state8_max < 1e-3 and compat["pass"] and state_check["all_pass"] and cam_check["pass"]),
                    "model_compatibility_pass": bool(compat["pass"]),
                    "physical_state_compare_pass": bool(state_check["all_pass"]),
                    "camera_compare_pass": bool(cam_check["pass"]),
                    "model_compatibility": compat["checks"], "state8_max_abs": state8_max,
                    "shift_snapshot_digest": digest(snap), "clean_snapshot_digest": digest(snap_clean),
                    "shift_model_digest": digest_model(snap), "clean_model_digest": digest_model(snap_clean),
                    "shift_controller_digest": digest_controller(snap), "clean_controller_digest": digest_controller(snap_clean),
                    "controller_compare_pass": bool(state_check["checks"].get("controller", {}).get("pass", False)),
                    "z0_sha256": zhash,
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
                    "z0": z0[0].float().cpu().numpy().tolist(),
                    "shift_success": bool(success),
                })
                for k in range(min(args.executed_k, ctrl_s.shape[1])):
                    action = np.clip(ctrl_s[0, k].cpu().numpy(), -1.0, 1.0)
                    obs_s, reward, done, info = shift.step(action); steps += 1
                    success = is_success(shift, info, reward, done)
                    if success or done or steps >= args.max_steps: break
                chunk += 1
            print(json.dumps({"state_id": state_id, "chunks": chunk, "success": bool(success), "steps": steps}), flush=True)
        finally:
            shift.close(); clean.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "action_consistency_pairs_v1", "suite": args.suite,
                                    "lighting_id": args.lighting_id, "role": args.role,
                                    "seed": args.seed, "max_chunks": args.max_chunks,
                                    "records": records}, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "n": len(records)}))


if __name__ == "__main__":
    main()
