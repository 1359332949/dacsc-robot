"""Action-distance validation for source-calibrated Target-only Mean.

This is a counterfactual action probe, not a closed-loop success evaluation.
The environment follows correctly unnormalized Frozen CoRP actions. At each
decision state, all candidates share the same flow-noise initialization.

The source-content-filtered candidate is experimental and is not a validated
TTA method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos/CoRP"))
sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))

from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation  # noqa: E402
from dacsc.state_pairing import model_compatible, snapshot  # noqa: E402
from dacsc.observation_render import render_policy_observation  # noqa: E402
from dacsc.action_guard import limit_rotation_delta  # noqa: E402
from dacsc.target_only_mean import (  # noqa: E402
    ConditionalSourceBaseline,
    TargetOnlyMeanEstimator,
    fit_source_calibration,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gate_diagnostics_json(diagnostics: dict) -> dict:
    """Convert one-item gate diagnostics to JSON-safe scalar values."""
    result = {}
    for key, value in diagnostics.items():
        if key == "abstain_reasons":
            result[key] = value[0]
        elif torch.is_tensor(value):
            flat = value.detach().float().cpu().reshape(-1)
            result[key] = float(flat[0]) if flat.numel() == 1 else flat.tolist()
        else:
            result[key] = value
    return result


def configure(args):
    import os
    import yaml

    src = args.libero_src.expanduser().resolve()
    root = src / "libero" / "libero"
    if not root.is_dir():
        raise FileNotFoundError(f"LIBERO source not found: {src}")
    sys.path.insert(0, str(src))
    cfg = args.libero_config.expanduser().resolve()
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "datasets").mkdir(exist_ok=True)
    cp = cfg / "config.yaml"
    if not cp.is_file():
        cp.write_text(yaml.safe_dump({
            "benchmark_root": str(root),
            "bddl_files": str(root / "bddl_files"),
            "init_states": str(root / "init_files"),
            "assets": str(root / "assets"),
            "datasets": str(cfg / "datasets"),
        }))
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


@torch.no_grad()
def sample_z(policy, representation, z0):
    gen = policy.action_generator
    action = z0.to(
        device=representation.tokens.device,
        dtype=representation.action_dtype,
    ).clone()
    prefix = gen.prepare_prefix(representation)
    for step in range(gen.num_sample_steps):
        tau = torch.full(
            (action.shape[0],),
            step / float(gen.num_sample_steps),
            device=action.device,
            dtype=action.dtype,
        )
        action = action + gen(action, tau, prefix) * (1.0 / float(gen.num_sample_steps))
    return action


@torch.no_grad()
def visual_from_obs(policy, obs):
    batch = prepare_observation(
        obs,
        config=policy.config,
        normalizer=policy.normalizer,
        device=next(policy.parameters()).device,
    )
    ext = policy.representation_extractor
    views = tuple(ext.view_names)
    patches = ext.visual_encoder(torch.cat([batch["images"][v] for v in views], dim=0))
    batch_size = batch["images"][views[0]].shape[0]
    patch_dict = {
        v: patches[i * batch_size:(i + 1) * batch_size]
        for i, v in enumerate(views)
    }
    language_tokens, pooled_language = ext.language_encoder(batch["instruction"])
    visual = ext.cross_attention_resampler(patch_dict, pooled_language)
    state_token = ext.state_encoder(batch["state"])
    representation = SimpleNamespace(
        tokens=torch.cat([visual, language_tokens.to(visual.dtype), state_token], dim=1),
        pooled_language=pooled_language,
        action_dtype=visual.dtype,
    )
    return visual, representation


def _rotation_geodesic(pred_rotvec: np.ndarray, ref_rotvec: np.ndarray) -> float:
    pred = Rotation.from_rotvec(pred_rotvec.reshape(-1, 3))
    ref = Rotation.from_rotvec(ref_rotvec.reshape(-1, 3))
    return float((pred.inv() * ref).magnitude().mean())


def _one_action_metrics(policy, pred_norm, ref_norm, horizon: int) -> dict:
    pred_norm = pred_norm[:, :horizon].float()
    ref_norm = ref_norm[:, :horizon].float()
    normalized_mae = float((pred_norm - ref_norm).abs().mean())

    pred_ctrl = policy.normalizer.unnormalize_action(pred_norm).float()
    ref_ctrl = policy.normalizer.unnormalize_action(ref_norm).float()
    pred_np = pred_ctrl.detach().cpu().numpy()
    ref_np = ref_ctrl.detach().cpu().numpy()
    return {
        "normalized_mae": normalized_mae,
        "xyz_mae_controller_units": float(np.abs(pred_np[..., :3] - ref_np[..., :3]).mean()),
        "rotation_geodesic_rad": _rotation_geodesic(pred_np[..., 3:6], ref_np[..., 3:6]),
        "gripper_mae": float(np.abs(pred_np[..., 6] - ref_np[..., 6]).mean()),
    }


def action_metrics(policy, pred_norm, ref_norm) -> dict:
    horizon = pred_norm.shape[1]
    return {
        "full": _one_action_metrics(policy, pred_norm, ref_norm, horizon),
        "prefix6": _one_action_metrics(policy, pred_norm, ref_norm, min(6, horizon)),
        "prefix12": _one_action_metrics(policy, pred_norm, ref_norm, min(12, horizon)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trajectories", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--clip-norm", type=float, default=20.0)
    ap.add_argument("--content-rank", type=int, default=2)
    ap.add_argument("--source-folds", type=int, default=5)
    ap.add_argument("--fixed-application-confidence", type=float, default=None,
                    help="optional fixed strength for the gated Target correction")
    ap.add_argument("--rotation-budget-rad", type=float, default=0.0,
                    help="SO(3) action-space trust region relative to Frozen; 0 preserves Frozen rotations")
    ap.add_argument("--min-update-confidence", type=float, default=None,
                    help="optional diagnostic override; default is maximum source-OOF calibration confidence")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    args = ap.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.out}")
    if args.trajectories < 1 or args.chunks < 1:
        raise ValueError("trajectories and chunks must be positive")

    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils

    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == "libero_object")
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == 2222)
    base_bddl = variant["base_bddl"]
    shift_bddl = variant["env_bddl_name"]
    language = BDDLUtils.get_problem_info(base_bddl)["language_instruction"]

    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device)
    policy.eval()
    horizon = int(policy.config["action_horizon"])
    if not (1 <= args.executed_k <= horizon):
        raise ValueError(f"executed-k must be in [1,{horizon}]")

    # Source calibration uses only Clean rows from the source development split.
    manifest = json.loads((args.capture / "manifest.json").read_text())
    features_path = args.capture / "features.npz"
    z = np.load(features_path)
    rows = [r for r in manifest["records"] if r.get("compare_pass")]
    by_state = {int(r["index"]): int(r["feature_row"]) for r in rows}
    dev = [int(i) for i in manifest["dev_indices"] if int(i) in by_state]
    fit_ids = dev[10:20]
    if len(fit_ids) < args.source_folds:
        raise ValueError("not enough source development states for requested cross-fit folds")

    def captured_pooled(state: int, kind: str) -> np.ndarray:
        patches = {
            view: torch.as_tensor(
                z[f"{kind}_{view}"][by_state[state]],
                device=device,
                dtype=torch.float32,
            ).unsqueeze(0)
            for view in manifest["policy_views"]
        }
        _, pooled_language = policy.representation_extractor.language_encoder([language])
        visual = policy.representation_extractor.cross_attention_resampler(
            patches, pooled_language
        )
        return visual.mean(dim=1)[0].detach().float().cpu().numpy()

    clean_fit = {state: captured_pooled(state, "clean") for state in fit_ids}
    shift_fit = {state: captured_pooled(state, "shift") for state in fit_ids}
    source_context = np.stack([z["state8"][by_state[state]] for state in fit_ids])
    source_clean_visual = np.stack([clean_fit[state] for state in fit_ids])

    source, content_projector, confidence_gate, source_diagnostics = fit_source_calibration(
        source_context,
        source_clean_visual,
        l2=args.l2,
        content_rank=args.content_rank,
        folds=args.source_folds,
        seed=args.seed,
    )
    # Privileged diagnostic reference only; never used by target estimators.
    paired_fit_mean = np.mean(
        [shift_fit[state] - clean_fit[state] for state in fit_ids], axis=0
    )

    held = [
        int(state)
        for state in task["state_split"]["heldout_indices"][:args.trajectories]
    ]
    overlap = sorted(set(held).intersection(fit_ids))
    if overlap:
        raise RuntimeError(f"held-out target states overlap source fit states: {overlap}")
    initial_states = torch.load(task["state_split"]["source_file"], weights_only=False)

    records = []
    for init_index in held:
        # Fresh variant environment per episode; do not reset away Lighting assets.
        env_shift = OffScreenRenderEnv(
            bddl_file_name=shift_bddl,
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=args.render_gpu,
        )
        env_clean = OffScreenRenderEnv(
            bddl_file_name=base_bddl,
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=args.render_gpu,
        )
        try:
            compatibility = model_compatible(
                snapshot(env_shift),
                snapshot(env_clean),
                strict_physics=True,
            )
            if not compatibility["pass"]:
                raise RuntimeError(
                    "Clean/Shift simulator models are not physics-compatible: "
                    f"{compatibility['checks']}"
                )

            env_shift.seed(args.seed + init_index)
            obs = env_shift.set_init_state(initial_states[init_index])
            done = False
            for _ in range(SETTLING_STEPS):
                obs, _, done, _ = env_shift.step(DUMMY_ACTION)
                if done:
                    raise RuntimeError(f"episode ended during settling: state={init_index}")

            # Ungated branches reproduce the existing estimator as diagnostics.
            target_estimator = TargetOnlyMeanEstimator(
                source, momentum=0.5, alpha=args.alpha, clip_norm=args.clip_norm
            )
            filtered_estimator = TargetOnlyMeanEstimator(
                source, momentum=0.5, alpha=args.alpha, clip_norm=args.clip_norm,
                residual_projector=content_projector,
            )
            # Source-only confidence gate abstains on source-like residuals and
            # contexts outside source calibration support.
            target_gated_estimator = TargetOnlyMeanEstimator(
                source, momentum=0.5, alpha=args.alpha, clip_norm=args.clip_norm,
                confidence_gate=confidence_gate,
                min_update_confidence=args.min_update_confidence,
            )
            filtered_gated_estimator = TargetOnlyMeanEstimator(
                source, momentum=0.5, alpha=args.alpha, clip_norm=args.clip_norm,
                residual_projector=content_projector, confidence_gate=confidence_gate,
                min_update_confidence=args.min_update_confidence,
            )
            # Separate Clean-stream branches are offline negative controls.
            clean_estimator = TargetOnlyMeanEstimator(
                source, momentum=0.5, alpha=args.alpha, clip_norm=args.clip_norm,
                confidence_gate=confidence_gate,
                min_update_confidence=args.min_update_confidence,
            )
            clean_filtered_estimator = TargetOnlyMeanEstimator(
                source, momentum=0.5, alpha=args.alpha, clip_norm=args.clip_norm,
                residual_projector=content_projector, confidence_gate=confidence_gate,
                min_update_confidence=args.min_update_confidence,
            )

            for chunk_index in range(args.chunks):
                state_snapshot = snapshot(env_shift)
                shift_policy_obs = policy_observation(obs, language, policy)
                visual_shift, rep_shift = visual_from_obs(policy, shift_policy_obs)

                # Offline Clean counterfactual at the exact same simulator state.
                clean_obs = render_policy_observation(env_clean, state_snapshot)
                clean_policy_obs = policy_observation(clean_obs, language, policy)
                visual_clean, rep_clean = visual_from_obs(policy, clean_policy_obs)
                state_error = float(
                    (clean_policy_obs["state"] - shift_policy_obs["state"]).abs().max()
                )
                # Twin rendering can differ at the 1e-3 proprio level after
                # executed chunks because the controller state is not part of
                # the MuJoCo snapshot; reject only larger mismatches and log it.
                if state_error > 5e-3:
                    raise RuntimeError(
                        f"Clean/Shift proprio mismatch at state={init_index}, "
                        f"chunk={chunk_index}: {state_error}"
                    )

                context = shift_policy_obs["state"].detach().cpu().numpy()
                pooled_shift = visual_shift.mean(dim=1)
                pooled_clean = visual_clean.mean(dim=1)

                raw_target_residual = target_estimator.raw_residual(pooled_shift, context)
                target_mean = target_estimator.update(pooled_shift, context)
                target_correction = target_estimator.applied_correction()
                filtered_target_residual = filtered_estimator.residual(pooled_shift, context)
                filtered_mean = filtered_estimator.update(pooled_shift, context)
                filtered_correction = filtered_estimator.applied_correction()

                target_gated_mean = target_gated_estimator.update(pooled_shift, context)
                target_gated_correction = target_gated_estimator.applied_correction()
                filtered_gated_mean = filtered_gated_estimator.update(pooled_shift, context)
                filtered_gated_correction = filtered_gated_estimator.applied_correction()

                # Clean residuals are offline negative controls only. The Clean
                # estimators consume their own Clean stream and separate EMA.
                clean_raw_residual = target_estimator.raw_residual(pooled_clean, context)
                clean_filtered_residual = content_projector.transform(clean_raw_residual)
                clean_mean = clean_estimator.update(pooled_clean, context)
                clean_correction = clean_estimator.applied_correction()
                clean_filtered_mean = clean_filtered_estimator.update(pooled_clean, context)
                clean_filtered_correction = clean_filtered_estimator.applied_correction()

                # V5: update acceptance remains residual novelty x context support,
                # while application is gated by the current observation's residual
                # novelty only. Context support is intentionally update-only because
                # later task states may leave source-state support under valid rollout.
                target_residual_application_confidence = (
                    target_gated_estimator.last_gate_diagnostics["residual_confidence"]
                )
                filtered_residual_application_confidence = (
                    filtered_gated_estimator.last_gate_diagnostics["residual_confidence"]
                )
                clean_residual_application_confidence = (
                    clean_estimator.last_gate_diagnostics["residual_confidence"]
                )
                target_gated_residual_apply_correction = (
                    target_gated_estimator.applied_correction(
                        application_confidence=target_residual_application_confidence
                    )
                )
                filtered_gated_residual_apply_correction = (
                    filtered_gated_estimator.applied_correction(
                        application_confidence=filtered_residual_application_confidence
                    )
                )
                # Offline paired-Clean negative control: evaluate the same stored
                # Target correction under Clean's current residual-novelty score.
                target_gated_clean_residual_apply_correction = (
                    target_gated_estimator.applied_correction(
                        application_confidence=clean_residual_application_confidence
                    )
                )
                filtered_gated_clean_residual_apply_correction = (
                    filtered_gated_estimator.applied_correction(
                        application_confidence=clean_residual_application_confidence
                    )
                )
                fixed_application_confidence = None
                target_gated_fixed_apply_correction = None
                filtered_gated_fixed_apply_correction = None
                target_gated_clean_fixed_apply_correction = None
                filtered_gated_clean_fixed_apply_correction = None
                if args.fixed_application_confidence is not None:
                    if not (0.0 <= args.fixed_application_confidence <= 1.0):
                        raise ValueError("--fixed-application-confidence must be in [0,1]")
                    fixed_application_confidence = torch.full_like(
                        target_gated_estimator.last_confidence,
                        args.fixed_application_confidence,
                    )
                    target_gated_fixed_apply_correction = target_gated_estimator.applied_correction(
                        application_confidence=fixed_application_confidence)
                    filtered_gated_fixed_apply_correction = filtered_gated_estimator.applied_correction(
                        application_confidence=fixed_application_confidence)
                    target_gated_clean_fixed_apply_correction = target_gated_estimator.applied_correction(
                        application_confidence=fixed_application_confidence)
                    filtered_gated_clean_fixed_apply_correction = filtered_gated_estimator.applied_correction(
                        application_confidence=fixed_application_confidence)

                paired_correction = torch.as_tensor(
                    paired_fit_mean, device=device, dtype=visual_shift.dtype
                ).view(1, 1, -1)
                target_correction = target_correction.to(
                    device=device, dtype=visual_shift.dtype
                ).view(1, 1, -1)
                filtered_correction = filtered_correction.to(
                    device=device, dtype=visual_shift.dtype
                ).view(1, 1, -1)
                target_gated_correction = target_gated_correction.to(
                    device=device, dtype=visual_shift.dtype
                ).view(1, 1, -1)
                filtered_gated_correction = filtered_gated_correction.to(
                    device=device, dtype=visual_shift.dtype
                ).view(1, 1, -1)
                target_gated_residual_apply_correction = (
                    target_gated_residual_apply_correction.to(
                        device=device, dtype=visual_shift.dtype
                    ).view(1, 1, -1)
                )
                filtered_gated_residual_apply_correction = (
                    filtered_gated_residual_apply_correction.to(
                        device=device, dtype=visual_shift.dtype
                    ).view(1, 1, -1)
                )
                target_gated_clean_residual_apply_correction = (
                    target_gated_clean_residual_apply_correction.to(
                        device=device, dtype=visual_clean.dtype
                    ).view(1, 1, -1)
                )
                filtered_gated_clean_residual_apply_correction = (
                    filtered_gated_clean_residual_apply_correction.to(
                        device=device, dtype=visual_clean.dtype
                    ).view(1, 1, -1)
                )
                if target_gated_fixed_apply_correction is not None:
                    target_gated_fixed_apply_correction = target_gated_fixed_apply_correction.to(
                        device=device, dtype=visual_shift.dtype).view(1, 1, -1)
                    filtered_gated_fixed_apply_correction = filtered_gated_fixed_apply_correction.to(
                        device=device, dtype=visual_shift.dtype).view(1, 1, -1)
                    target_gated_clean_fixed_apply_correction = target_gated_clean_fixed_apply_correction.to(
                        device=device, dtype=visual_clean.dtype).view(1, 1, -1)
                    filtered_gated_clean_fixed_apply_correction = filtered_gated_clean_fixed_apply_correction.to(
                        device=device, dtype=visual_clean.dtype).view(1, 1, -1)
                clean_correction = clean_correction.to(
                    device=device, dtype=visual_clean.dtype
                ).view(1, 1, -1)
                clean_filtered_correction = clean_filtered_correction.to(
                    device=device, dtype=visual_clean.dtype
                ).view(1, 1, -1)

                z0_seed = int(args.seed + 100000 * init_index + chunk_index)
                generator = torch.Generator(device=device)
                generator.manual_seed(z0_seed)
                z0 = torch.randn(
                    1,
                    policy.action_generator.action_horizon,
                    policy.action_generator.action_dim,
                    device=device,
                    dtype=rep_shift.action_dtype,
                    generator=generator,
                )

                def act(visual: torch.Tensor, base_rep) -> torch.Tensor:
                    rep = SimpleNamespace(
                        tokens=torch.cat(
                            [visual, base_rep.tokens[:, visual.shape[1]:]], dim=1
                        ),
                        pooled_language=base_rep.pooled_language,
                        action_dtype=visual.dtype,
                    )
                    return sample_z(policy, rep, z0)

                action_clean = act(visual_clean, rep_clean)
                action_frozen = act(visual_shift, rep_shift)
                action_paired = act(visual_shift - paired_correction, rep_shift)
                action_target = act(visual_shift - target_correction, rep_shift)
                action_filtered = act(visual_shift - filtered_correction, rep_shift)
                action_target_gated = act(visual_shift - target_gated_correction, rep_shift)
                action_filtered_gated = act(visual_shift - filtered_gated_correction, rep_shift)
                action_target_gated_residual_apply = act(
                    visual_shift - target_gated_residual_apply_correction, rep_shift
                )
                action_filtered_gated_residual_apply = act(
                    visual_shift - filtered_gated_residual_apply_correction, rep_shift
                )
                if target_gated_fixed_apply_correction is not None:
                    action_target_gated_fixed_apply = act(
                        visual_shift - target_gated_fixed_apply_correction, rep_shift)
                    action_filtered_gated_fixed_apply = act(
                        visual_shift - filtered_gated_fixed_apply_correction, rep_shift)
                action_target_gated_guarded = limit_rotation_delta(
                    action_target_gated, action_frozen, policy.normalizer,
                    args.rotation_budget_rad,
                )
                action_filtered_gated_guarded = limit_rotation_delta(
                    action_filtered_gated, action_frozen, policy.normalizer,
                    args.rotation_budget_rad,
                )
                action_reverse = act(visual_shift + target_correction, rep_shift)

                # Shift-derived correction applied to Clean is a cross-domain
                # transfer stress test, not operational Clean-only retention.
                clean_target_cost = act(visual_clean - target_correction, rep_clean)
                clean_filtered_cost = act(visual_clean - filtered_correction, rep_clean)
                clean_target_gated_transfer = act(
                    visual_clean - target_gated_correction, rep_clean
                )
                clean_filtered_gated_transfer = act(
                    visual_clean - filtered_gated_correction, rep_clean
                )
                clean_target_gated_residual_apply_transfer = act(
                    visual_clean - target_gated_clean_residual_apply_correction, rep_clean
                )
                clean_filtered_gated_residual_apply_transfer = act(
                    visual_clean - filtered_gated_clean_residual_apply_correction, rep_clean
                )
                if target_gated_fixed_apply_correction is not None:
                    clean_target_gated_fixed_apply_transfer = act(
                        visual_clean - target_gated_clean_fixed_apply_correction, rep_clean)
                    clean_filtered_gated_fixed_apply_transfer = act(
                        visual_clean - filtered_gated_clean_fixed_apply_correction, rep_clean)
                # Operational Clean negative controls: independent Clean-stream EMA.
                clean_only_action = act(visual_clean - clean_correction, rep_clean)
                clean_filtered_only_action = act(
                    visual_clean - clean_filtered_correction, rep_clean
                )

                # log the twin error below
                records.append({
                    "init_index": init_index,
                    "chunk": chunk_index,
                    "z0_seed": z0_seed,
                    "state_pair_max_abs_error": state_error,
                    "frozen": action_metrics(policy, action_frozen, action_clean),
                    "paired_fit_mean": action_metrics(policy, action_paired, action_clean),
                    "target_only_mean": action_metrics(policy, action_target, action_clean),
                    "target_only_content_filtered": action_metrics(
                        policy, action_filtered, action_clean
                    ),
                    "target_only_gated": action_metrics(
                        policy, action_target_gated, action_clean
                    ),
                    "target_only_content_filtered_gated": action_metrics(
                        policy, action_filtered_gated, action_clean
                    ),
                    "target_only_gated_residual_apply": action_metrics(
                        policy, action_target_gated_residual_apply, action_clean
                    ),
                    "target_only_content_filtered_gated_residual_apply": action_metrics(
                        policy, action_filtered_gated_residual_apply, action_clean
                    ),
                    **({"target_only_gated_fixed_apply": action_metrics(
                        policy, action_target_gated_fixed_apply, action_clean),
                        "target_only_content_filtered_gated_fixed_apply": action_metrics(
                        policy, action_filtered_gated_fixed_apply, action_clean)}
                       if target_gated_fixed_apply_correction is not None else {}),
                    "target_only_gated_rotation_guarded": action_metrics(
                        policy, action_target_gated_guarded, action_clean
                    ),
                    "target_only_content_filtered_gated_rotation_guarded": action_metrics(
                        policy, action_filtered_gated_guarded, action_clean
                    ),
                    "target_only_reverse": action_metrics(
                        policy, action_reverse, action_clean
                    ),
                    "oracle_clean_replacement": action_metrics(
                        policy, action_clean, action_clean
                    ),
                    "clean_cost_target_estimate": action_metrics(
                        policy, clean_target_cost, action_clean
                    ),
                    "clean_cost_filtered_target_estimate": action_metrics(
                        policy, clean_filtered_cost, action_clean
                    ),
                    "clean_cost_target_gated_transfer": action_metrics(
                        policy, clean_target_gated_transfer, action_clean
                    ),
                    "clean_cost_filtered_gated_transfer": action_metrics(
                        policy, clean_filtered_gated_transfer, action_clean
                    ),
                    "clean_cost_gated_residual_apply_transfer": action_metrics(
                        policy, clean_target_gated_residual_apply_transfer, action_clean
                    ),
                    "clean_cost_filtered_gated_residual_apply_transfer": action_metrics(
                        policy, clean_filtered_gated_residual_apply_transfer, action_clean
                    ),
                    **({"clean_cost_gated_fixed_apply_transfer": action_metrics(
                        policy, clean_target_gated_fixed_apply_transfer, action_clean),
                        "clean_cost_filtered_gated_fixed_apply_transfer": action_metrics(
                        policy, clean_filtered_gated_fixed_apply_transfer, action_clean)}
                       if target_gated_fixed_apply_correction is not None else {}),
                    "clean_only_estimate_negative_control": action_metrics(
                        policy, clean_only_action, action_clean
                    ),
                    "clean_only_filtered_negative_control": action_metrics(
                        policy, clean_filtered_only_action, action_clean
                    ),
                    "target_raw_residual_norm": float(raw_target_residual.norm()),
                    "target_ema_residual_norm": float(target_mean.norm()),
                    "target_filtered_residual_norm": float(filtered_mean.norm()),
                    "target_gated_correction_norm": float(target_gated_correction.norm()),
                    "target_filtered_gated_correction_norm": float(filtered_gated_correction.norm()),
                    "target_gated_residual_apply_correction_norm": float(
                        target_gated_residual_apply_correction.norm()
                    ),
                    "target_filtered_gated_residual_apply_correction_norm": float(
                        filtered_gated_residual_apply_correction.norm()
                    ),
                    "target_residual_application_confidence": float(
                        target_residual_application_confidence.mean()
                    ),
                    "clean_residual_application_confidence": float(
                        clean_residual_application_confidence.mean()
                    ),
                    "target_confidence": float(target_gated_estimator.last_confidence.mean()),
                    "filtered_confidence": float(filtered_gated_estimator.last_confidence.mean()),
                    "clean_confidence": float(clean_estimator.last_confidence.mean()),
                    "clean_filtered_confidence": float(clean_filtered_estimator.last_confidence.mean()),
                    "target_gate_diagnostics": gate_diagnostics_json(
                        target_gated_estimator.last_gate_diagnostics
                    ),
                    "filtered_gate_diagnostics": gate_diagnostics_json(
                        filtered_gated_estimator.last_gate_diagnostics
                    ),
                    "clean_gate_diagnostics": gate_diagnostics_json(
                        clean_estimator.last_gate_diagnostics
                    ),
                    "target_update_accepted": bool(
                        target_gated_estimator.last_update_accepted.all()
                    ),
                    "filtered_update_accepted": bool(
                        filtered_gated_estimator.last_update_accepted.all()
                    ),
                    "clean_update_accepted": bool(
                        clean_estimator.last_update_accepted.all()
                    ),
                    "clean_raw_residual_norm": float(clean_raw_residual.norm()),
                    "clean_filtered_residual_norm": float(clean_filtered_residual.norm()),
                    "target_clean_raw_gap_norm": float(
                        (raw_target_residual - clean_raw_residual).norm()
                    ),
                    "content_filter_retained_ratio": float(
                        filtered_target_residual.norm()
                        / raw_target_residual.norm().clamp_min(1e-12)
                    ),
                    "actual_executed_actions": 0,
                })

                # The sampler returns normalized actions. Match policy.predict():
                # unnormalize before sending controller commands to LIBERO.
                controller_chunk = policy.normalizer.unnormalize_action(
                    action_frozen
                )[0].float().detach().cpu().numpy()
                actual_executed = 0
                for action in controller_chunk[:args.executed_k]:
                    obs, reward, done, info = env_shift.step(
                        np.clip(action, -1.0, 1.0)
                    )
                    actual_executed += 1
                    if done:
                        break
                records[-1]["actual_executed_actions"] = actual_executed
                if done or actual_executed != args.executed_k:
                    break
        finally:
            env_shift.close()
            env_clean.close()

    if not records:
        raise RuntimeError("action validation produced no decision-state records")

    method_names = (
        "frozen",
        "paired_fit_mean",
        "target_only_mean",
        "target_only_content_filtered",
        "target_only_gated",
        "target_only_content_filtered_gated",
        "target_only_gated_residual_apply",
        "target_only_content_filtered_gated_residual_apply",
        *(["target_only_gated_fixed_apply", "target_only_content_filtered_gated_fixed_apply"]
          if args.fixed_application_confidence is not None else []),
        "target_only_gated_rotation_guarded",
        "target_only_content_filtered_gated_rotation_guarded",
        "target_only_reverse",
        "oracle_clean_replacement",
    )
    clean_control_names = (
        "clean_cost_target_estimate",
        "clean_cost_filtered_target_estimate",
        "clean_cost_target_gated_transfer",
        "clean_cost_filtered_gated_transfer",
        "clean_cost_gated_residual_apply_transfer",
        "clean_cost_filtered_gated_residual_apply_transfer",
        *(["clean_cost_gated_fixed_apply_transfer",
            "clean_cost_filtered_gated_fixed_apply_transfer"]
          if args.fixed_application_confidence is not None else []),
        "clean_only_estimate_negative_control",
        "clean_only_filtered_negative_control",
    )
    metric_names = (
        "normalized_mae",
        "xyz_mae_controller_units",
        "rotation_geodesic_rad",
        "gripper_mae",
    )
    summary = {}
    for method in method_names:
        summary[method] = {}
        for horizon_name in ("full", "prefix6", "prefix12"):
            summary[method][horizon_name] = {
                metric: float(np.mean([
                    record[method][horizon_name][metric] for record in records
                ]))
                for metric in metric_names
            }

    per_episode_summary = {}
    all_names = method_names + clean_control_names
    for episode_id in sorted({record["init_index"] for record in records}):
        episode_records = [r for r in records if r["init_index"] == episode_id]
        per_episode_summary[str(episode_id)] = {}
        for method in all_names:
            per_episode_summary[str(episode_id)][method] = {}
            for horizon_name in ("full", "prefix6", "prefix12"):
                per_episode_summary[str(episode_id)][method][horizon_name] = {
                    metric: float(np.mean([
                        r[method][horizon_name][metric] for r in episode_records
                    ]))
                    for metric in metric_names
                }

    clean_control_summary = {}
    for method in clean_control_names:
        clean_control_summary[method] = {}
        for horizon_name in ("full", "prefix6", "prefix12"):
            clean_control_summary[method][horizon_name] = {
                metric: float(np.mean([
                    r[method][horizon_name][metric] for r in records
                ]))
                for metric in metric_names
            }

    payload = {
        "schema": "target_only_action_validation_v5",
        "status": "completed",
        "claim_scope": "counterfactual action-distance validation only; not closed-loop SR",
        "condition": "Object x Lighting 2222",
        "fit_ids": fit_ids,
        "heldout_ids": held,
        "source_calibration": source_diagnostics,
        "confidence_gate": {
            "update_type": "source-oof residual novelty multiplied by nearest-source-context support",
            "application_type": "current-observation source residual novelty only; context support is update-only",
            "min_update_confidence_override": args.min_update_confidence,
            "default_min_update_confidence": source_diagnostics["confidence_gate"]["source_calibrated_min_update_confidence"],
            "update_semantics": "only observations above the source-OOF calibration confidence maximum update episode EMA; rejected observations hold last accepted estimate",
            "logged_components": [
                "residual_distance", "residual_threshold", "residual_confidence",
                "nearest_context_distance", "context_support_radius",
                "context_support_outer_radius", "context_confidence",
                "combined_confidence", "abstain_reasons", "update_accepted",
            ],
            "interpretation": "source-only update-abstention heuristic; appearance estimate is held across later OOD contexts",
            "application_semantics": "the held correction is scaled per observation by residual_confidence without reusing the context-support term",
            "fixed_application_confidence": args.fixed_application_confidence,
        },
        "rotation_trust_region": {
            "max_relative_so3_change_rad": args.rotation_budget_rad,
            "zero_budget_preserves_frozen_rotation": args.rotation_budget_rad == 0.0,
        },
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "checkpoint_sha256": sha256(args.checkpoint),
        "capture_manifest_sha256": sha256(args.capture / "manifest.json"),
        "capture_features_sha256": sha256(features_path),
        "information_budget": {
            "source_baseline": "source development Clean features and observable proprioception only",
            "source_content_projector": "cross-fitted source-Clean residuals only",
            "target_estimators": "target Shift observations/history and proprioception only",
            "paired_fit_mean": "privileged fit-state Clean/Shift diagnostic reference",
            "oracle_clean_replacement": "privileged same-state Clean observation; upper-bound diagnostic",
            "clean_residuals": "offline negative control only; never fed to target estimators",
        },
        "environment_protocol": {
            "fresh_shift_env_per_episode": True,
            "shift_env_reset_called": False,
            "clean_shift_physics_compatibility_checked": True,
            "executed_policy": "Frozen CoRP",
            "executed_action_conversion": "normalizer.unnormalize_action then clip to controller range",
        },
        "records": records,
        "record_count": len(records),
        "episode_count": len(per_episode_summary),
        "summary": summary,
        "clean_control_summary": clean_control_summary,
        "per_episode_summary": per_episode_summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "summary": summary,
        "clean_control_summary": clean_control_summary,
        "per_episode_summary": per_episode_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
