"""Collect strict source-only Clean/Shift DINO patch pairs for offline diagnostics.

The collector records raw per-view visual_encoder outputs (before the
CrossAttentionResampler), matched physical states, state inputs, and one
explicit shared z0. It does not collect target-dev or target-test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos/CoRP"))
sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))

from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from corp.representation.extractor import Representation  # noqa: E402
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa: E402
from target_only_action_validation import configure  # noqa: E402
from dacsc.differentiable_policy import sample_from_z0  # noqa: E402
from dacsc.state_pairing import (  # noqa: E402
    snapshot, restore, model_compatible, compare, camera_compare,
    digest, digest_model, digest_controller,
)
from dacsc.tta_protocol import load_split, role_ids, validate_split_identity  # noqa: E402
from dacsc.patch_diagnostics import validate_patch_dict  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_rev(path: Path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def encode_observation(policy, observation, z0):
    """Return pre-Resampler patches and the same frozen action path used downstream."""
    device = next(policy.parameters()).device
    prepared = prepare_observation(
        observation,
        config=policy.config,
        normalizer=policy.normalizer,
        device=device,
    )
    extractor = policy.representation_extractor
    views = tuple(extractor.view_names)
    image_batches = [prepared["images"][view] for view in views]
    batch_size = int(image_batches[0].shape[0])
    if any(int(x.shape[0]) != batch_size for x in image_batches):
        raise RuntimeError("per-view image batch sizes differ")
    visual_batch = extractor.visual_encoder(torch.cat(image_batches, dim=0))
    if visual_batch.ndim != 3 or visual_batch.shape[0] != len(views) * batch_size:
        raise RuntimeError(
            f"visual_encoder expected {len(views) * batch_size} [B,N,C] rows, "
            f"got {tuple(visual_batch.shape)}"
        )
    patches = {
        view: visual_batch[i * batch_size:(i + 1) * batch_size]
        for i, view in enumerate(views)
    }
    validate_patch_dict(patches, views)
    language_tokens, pooled_language = extractor.language_encoder(prepared["instruction"])
    visual_tokens = extractor.cross_attention_resampler(patches, pooled_language)
    state_token = extractor.state_encoder(prepared["state"])
    representation = Representation(
        tokens=torch.cat(
            [visual_tokens, language_tokens.to(visual_tokens.dtype), state_token], dim=1
        ),
        pooled_language=pooled_language,
        action_dtype=visual_tokens.dtype,
    )
    normalized_actions = sample_from_z0(policy, representation, z0)
    controller_actions = policy.normalizer.unnormalize_action(normalized_actions)
    return {
        "patches": patches,
        "visual_tokens": visual_tokens,
        "language_tokens": language_tokens,
        "pooled_language": pooled_language,
        "state_input": prepared["state"],
        "state_token": state_token,
        "normalized_actions": normalized_actions,
        "controller_actions": controller_actions,
        "representation": representation,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--phase0-report", type=Path, required=True,
                    help="passing Phase 0 report for these exact checkpoint/matrix/split inputs")
    ap.add_argument("--out", type=Path, required=True,
                    help="output .npz; a sibling .manifest.json is also written")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--roles", default="source_fit,source_validation",
                    help="comma-separated subset of source_fit,source_validation")
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--early-chunks", default="0,1")
    ap.add_argument("--late-chunks", default="6,7,8,9")
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()

    phase0_report = json.loads(args.phase0_report.read_text())
    if phase0_report.get("schema") != "corp_dino_patch_interface_v1":
        raise ValueError("unexpected Phase 0 report schema")
    if not phase0_report.get("identity_pass", False):
        raise RuntimeError("Phase 0 identity gate did not pass; refusing pair collection")
    if phase0_report.get("suite") != args.suite or int(phase0_report.get("lighting_id", -1)) != int(args.lighting_id):
        raise ValueError("Phase 0 suite/lighting does not match pair collection")
    expected_hashes = {
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "matrix_sha256": _sha256_file(args.matrix),
        "split_sha256": _sha256_file(args.split_json),
    }
    for key, value in expected_hashes.items():
        if phase0_report.get(key) != value:
            raise ValueError(f"Phase 0 {key} does not match pair collection input")
    current_repo_rev = _git_rev(ROOT)
    current_corp_rev = _git_rev(ROOT / "repos/CoRP")
    if phase0_report.get("repository_git_rev") and phase0_report["repository_git_rev"] != current_repo_rev:
        raise ValueError("repository revision changed after Phase 0; rerun Phase 0")
    if phase0_report.get("corp_git_rev") and phase0_report["corp_git_rev"] != current_corp_rev:
        raise ValueError("CoRP revision changed after Phase 0; rerun Phase 0")

    if args.out.suffix != ".npz":
        raise ValueError("--out must end in .npz")
    manifest_path = args.out.with_suffix(".manifest.json")
    if args.out.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.out} or {manifest_path}")
    if os.environ.get("MUJOCO_GL") not in (None, "osmesa"):
        raise RuntimeError("patch-pair collection requires MUJOCO_GL=osmesa")
    os.environ["MUJOCO_GL"] = "osmesa"
    configure(args)

    roles = [x.strip() for x in args.roles.split(",") if x.strip()]
    if not roles or any(x not in ("source_fit", "source_validation") for x in roles):
        raise ValueError("only source_fit/source_validation roles are permitted")
    early_chunks = {int(x) for x in args.early_chunks.split(",") if x.strip()}
    late_chunks = {int(x) for x in args.late_chunks.split(",") if x.strip()}
    if early_chunks & late_chunks:
        raise ValueError("early and late chunk sets must be disjoint")
    selected_chunks = early_chunks | late_chunks
    if not selected_chunks or min(selected_chunks) < 0:
        raise ValueError("at least one non-negative chunk index is required")
    max_chunk = max(selected_chunks)

    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils

    matrix = json.loads(args.matrix.read_text())
    task = next(x for x in matrix["base_tasks"] if x["suite"] == args.suite)
    split = load_split(args.split_json)
    validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    variant = next(
        v for v in task["variants"]["lighting"]
        if int(v["classification_id"]) == int(args.lighting_id)
    )
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    states = torch.load(task["state_split"]["source_file"], map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    arrays: dict[str, list[np.ndarray]] = {}
    records: list[dict] = []
    views = tuple(policy.representation_extractor.view_names)
    for role in roles:
        state_ids = role_ids(split, role)
        for state_id in state_ids:
            shift = OffScreenRenderEnv(
                bddl_file_name=variant["env_bddl_name"],
                camera_heights=256, camera_widths=256,
                render_gpu_device_id=args.render_gpu,
            )
            clean = OffScreenRenderEnv(
                bddl_file_name=variant["base_bddl"],
                camera_heights=256, camera_widths=256,
                render_gpu_device_id=args.render_gpu,
            )
            try:
                env_seed = int(args.seed)
                np.random.seed(env_seed)
                torch.manual_seed(env_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(env_seed)
                shift.seed(env_seed)
                shift.reset()
                obs_shift = shift.set_init_state(states[state_id])
                for _ in range(SETTLING_STEPS):
                    obs_shift, _, done, _ = shift.step(DUMMY_ACTION)
                    if done:
                        raise RuntimeError(f"episode ended during settling: {state_id}")

                step_count = 0
                chunk = 0
                success = False
                done = False
                while (not success and not done and step_count < args.max_steps
                       and chunk <= max_chunk):
                    po_shift = policy_observation(obs_shift, language, policy)
                    torch.manual_seed(env_seed + 100003 * int(state_id) + int(chunk))
                    z0 = torch.randn(
                        (1, policy.action_generator.action_horizon,
                         policy.action_generator.action_dim),
                        device=device, dtype=next(policy.parameters()).dtype,
                    )
                    shift_encoded = encode_observation(policy, po_shift, z0)
                    capture_window = (
                        "early" if chunk in early_chunks
                        else "late" if chunk in late_chunks
                        else None
                    )
                    if capture_window is not None:
                        snap_shift = snapshot(shift)
                        clean.seed(env_seed + 991 * chunk)
                        clean.reset()
                        clean.env.sim.model.body_pos[:] = np.asarray(
                            snap_shift["model_signature"]["body_pos"], dtype=float
                        )
                        clean.env.sim.model.body_quat[:] = np.asarray(
                            snap_shift["model_signature"]["body_quat"], dtype=float
                        )
                        clean.env.sim.forward()
                        compatibility = model_compatible(
                            snapshot(clean), snap_shift,
                            strict_bodies=True, strict_geometry=True, strict_physics=True,
                        )
                        if not compatibility["pass"]:
                            raise RuntimeError(
                                f"model compatibility failed: state={state_id}, "
                                f"chunk={chunk}, checks={compatibility['checks']}"
                            )
                        restore(clean, snap_shift)
                        clean._update_observables(force=True)
                        obs_clean = clean.env._get_observations()
                        snap_clean = snapshot(clean)
                        state_check = compare(
                            snap_shift, snap_clean,
                            tol={"body_xpos": 1e-3, "eef_pos": 1e-3,
                                 "body_xquat_rad": 2e-3},
                        )
                        camera_check = camera_compare(
                            snap_shift, snap_clean, expect_equal=True, tol=1e-3
                        )
                        po_clean = policy_observation(obs_clean, language, policy)
                        torch.manual_seed(env_seed + 100003 * int(state_id) + int(chunk))
                        clean_encoded = encode_observation(policy, po_clean, z0)

                        state_delta = float((
                            shift_encoded["state_input"].float()
                            - clean_encoded["state_input"].float()
                        ).abs().max().item())
                        controller_check = bool(
                            state_check["checks"].get("controller", {}).get("pass", False)
                        )
                        clean_patch_shapes = {
                            view: list(clean_encoded["patches"][view].shape)
                            for view in views
                        }
                        shift_patch_shapes = {
                            view: list(shift_encoded["patches"][view].shape)
                            for view in views
                        }
                        patch_shape_pass = bool(
                            all(clean_patch_shapes[v] == shift_patch_shapes[v] for v in views)
                            and clean_encoded["visual_tokens"].shape
                                == shift_encoded["visual_tokens"].shape
                        )
                        pairing_pass = bool(
                            compatibility["pass"] and state_check["all_pass"]
                            and camera_check["pass"] and state_delta < 1e-3
                            and controller_check and patch_shape_pass
                        )
                        if not pairing_pass:
                            # Retain the record for audit, but downstream scripts
                            # will exclude it from every quantitative summary.
                            pass

                        row = len(records)
                        for view in views:
                            arrays.setdefault(f"clean_{view}", []).append(
                                clean_encoded["patches"][view][0].float().cpu().numpy()
                            )
                            arrays.setdefault(f"shift_{view}", []).append(
                                shift_encoded["patches"][view][0].float().cpu().numpy()
                            )
                        for key, value in (
                            ("clean_state", clean_encoded["state_input"]),
                            ("shift_state", shift_encoded["state_input"]),
                            ("clean_state_token", clean_encoded["state_token"]),
                            ("shift_state_token", shift_encoded["state_token"]),
                            ("clean_language_tokens", clean_encoded["language_tokens"]),
                            ("shift_language_tokens", shift_encoded["language_tokens"]),
                            ("clean_pooled_language", clean_encoded["pooled_language"]),
                            ("shift_pooled_language", shift_encoded["pooled_language"]),
                            ("z0", z0),
                            ("clean_normalized_actions", clean_encoded["normalized_actions"]),
                            ("shift_normalized_actions", shift_encoded["normalized_actions"]),
                            ("clean_controller_actions", clean_encoded["controller_actions"]),
                            ("shift_controller_actions", shift_encoded["controller_actions"]),
                        ):
                            arrays.setdefault(key, []).append(
                                value[0].detach().float().cpu().numpy()
                            )
                        records.append({
                            "feature_row": row,
                            "state_id": int(state_id),
                            "role": role,
                            "window": capture_window,
                            "chunk_index": int(chunk),
                            "step": int(step_count),
                            "env_seed": env_seed,
                            "z0_sha256": hashlib.sha256(
                                z0.detach().float().cpu().numpy().tobytes()
                            ).hexdigest(),
                            "pairing_pass": pairing_pass,
                            "model_compatibility_pass": bool(compatibility["pass"]),
                            "physical_state_compare_pass": bool(state_check["all_pass"]),
                            "camera_compare_pass": bool(camera_check["pass"]),
                            "controller_compare_pass": controller_check,
                            "patch_shape_pass": patch_shape_pass,
                            "state_input_max_abs": state_delta,
                            "shift_snapshot_digest": digest(snap_shift),
                            "clean_snapshot_digest": digest(snap_clean),
                            "shift_model_digest": digest_model(snap_shift),
                            "clean_model_digest": digest_model(snap_clean),
                            "shift_controller_digest": digest_controller(snap_shift),
                            "clean_controller_digest": digest_controller(snap_clean),
                            "clean_patch_shapes": clean_patch_shapes,
                            "shift_patch_shapes": shift_patch_shapes,
                            "clean_visual_token_shape": list(clean_encoded["visual_tokens"].shape),
                            "shift_visual_token_shape": list(shift_encoded["visual_tokens"].shape),
                        })

                    controller = shift_encoded["controller_actions"]
                    for action_index in range(min(args.executed_k, controller.shape[1])):
                        action = np.clip(
                            controller[0, action_index].detach().cpu().numpy(), -1.0, 1.0
                        )
                        obs_shift, reward, done, info = shift.step(action)
                        step_count += 1
                        success = is_success(shift, info, reward, done)
                        if success or done or step_count >= args.max_steps:
                            break
                    chunk += 1
                print(json.dumps({
                    "role": role, "state_id": int(state_id), "chunks": int(chunk),
                    "success": bool(success), "steps": int(step_count),
                    "captured": sum(r["state_id"] == int(state_id) and r["role"] == role
                                    for r in records),
                }), flush=True)
            finally:
                shift.close()
                clean.close()

    if not records:
        raise RuntimeError("no patch pairs collected")
    output_arrays = {}
    for key, values in arrays.items():
        try:
            output_arrays[key] = np.stack(values)
        except ValueError as exc:
            raise RuntimeError(
                f"inconsistent array shape for {key}; inspect patch/input grid changes"
            ) from exc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **output_arrays)
    manifest = {
        "schema": "corp_paired_dino_patches_v1",
        "suite": args.suite,
        "lighting_id": int(args.lighting_id),
        "roles": roles,
        "windows": {"early_chunks": sorted(early_chunks), "late_chunks": sorted(late_chunks)},
        "env_seed": int(args.seed),
        "executed_k": int(args.executed_k),
        "records_total": len(records),
        "records_pairing_pass": sum(bool(r["pairing_pass"]) for r in records),
        "view_names": list(views),
        "patch_shapes": {
            view: list(output_arrays[f"shift_{view}"].shape[1:])
            for view in views
        },
        "feature_file": str(args.out),
        "feature_file_sha256": _sha256_file(args.out),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "matrix_sha256": _sha256_file(args.matrix),
        "split_sha256": _sha256_file(args.split_json),
        "initial_state_file_sha256": _sha256_file(Path(task["state_split"]["source_file"])),
        "phase0_report": str(args.phase0_report),
        "phase0_report_sha256": _sha256_file(args.phase0_report),
        "collector_script_sha256": _sha256_file(Path(__file__)),
        "repository_git_rev": _git_rev(ROOT),
        "corp_git_rev": _git_rev(ROOT / "repos/CoRP"),
        "libero_git_rev": _git_rev(args.libero_src),
        "records": records,
        "data_policy": (
            "source_fit/source_validation only; paired Clean patches are offline "
            "diagnostic interventions and are never a deployment input"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "feature_file": str(args.out),
        "manifest": str(manifest_path),
        "records_total": len(records),
        "records_pairing_pass": manifest["records_pairing_pass"],
        "patch_shapes": manifest["patch_shapes"],
    }, indent=2))


if __name__ == "__main__":
    main()
