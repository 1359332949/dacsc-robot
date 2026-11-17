"""Minimal closed-loop pilot: execute each method's own actions.

Protocol mirrors repos/CoRP/scripts/evaluate_libero.py:
- 10 settling steps with DUMMY_ACTION
- policy observations via the released policy_observation()
- action chunks unnormalized, then np.clip(action, -1, 1)
- official is_success(env, info, reward, done)
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos/CoRP"))
sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))

from evaluate_libero import (  # noqa: E402
    DUMMY_ACTION, MAX_STEPS, SETTLING_STEPS, is_success, policy_observation,
)
from corp import load_policy  # noqa: E402
from dacsc.target_only_mean import (  # noqa: E402
    ConditionalSourceBaseline, TargetOnlyMeanEstimator, fit_source_calibration,
)
from target_only_action_validation import configure, sample_z, visual_from_obs  # noqa: E402
from dacsc.observation_render import render_policy_observation  # noqa: E402
from dacsc.state_pairing import snapshot  # noqa: E402
from dacsc.pilot_validation import (  # noqa: E402
    ALLOWED_METHODS, parse_init_indices, validate_executed_k,
    validate_init_indices, validate_methods,
)


def build_source(args, policy, manifest, z, by_state, device, language):
    dev = [int(i) for i in manifest["dev_indices"] if int(i) in by_state]
    fit_ids = dev[args.source_fit_start:args.source_fit_start + args.source_fit_count]
    if len(fit_ids) < args.source_folds:
        raise ValueError("not enough source fit states")

    def captured_pooled(state):
        patches = {
            view: torch.as_tensor(z[f"clean_{view}"][by_state[state]], device=device,
                                  dtype=torch.float32).unsqueeze(0)
            for view in manifest["policy_views"]
        }
        _, pooled = policy.representation_extractor.language_encoder([language])
        visual = policy.representation_extractor.cross_attention_resampler(patches, pooled)
        return visual.mean(dim=1)[0].detach().float().cpu().numpy()

    context = np.stack([z["state8"][by_state[s]] for s in fit_ids])
    clean = np.stack([captured_pooled(s) for s in fit_ids])
    source, _, _, diag = fit_source_calibration(
        context, clean, l2=args.l2, content_rank=args.content_rank,
        folds=args.source_folds, seed=args.seed,
    )
    return fit_ids, source, diag


def make_rep(visual, base_rep):
    return SimpleNamespace(
        tokens=torch.cat([visual, base_rep.tokens[:, visual.shape[1]:]], dim=1),
        pooled_language=base_rep.pooled_language,
        action_dtype=visual.dtype,
    )


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def style_mean_only(patches, source_mean, strength=1.0):
    return patches + strength * (source_mean - patches.mean(dim=1, keepdim=True))


def style_transfer(patches, source_mean, source_std):
    mean_t = patches.mean(dim=1, keepdim=True)
    std_t = patches.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (patches - mean_t) / std_t * source_std + source_mean


def build_style_stats(args, policy, manifest, z, by_state, device, language):
    import numpy as np
    dev = [int(i) for i in manifest["dev_indices"] if int(i) in by_state]
    fit_ids = dev[args.source_fit_start:args.source_fit_start + args.source_fit_count]
    if len(fit_ids) != args.source_fit_count:
        raise ValueError(f"source fit returned {len(fit_ids)}, requested {args.source_fit_count}")
    views = list(manifest["policy_views"])
    mean, std = {}, {}
    for view in views:
        vals = torch.stack([
            torch.as_tensor(z[f"clean_{view}"][by_state[s]], dtype=torch.float32)
            for s in fit_ids
        ])
        mean[view] = vals.mean(dim=(0, 1), keepdim=True).to(device)
        std[view] = vals.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(1e-6).to(device)
    return fit_ids, mean, std


def score_leftover(candidate_actions, prev_leftover, executed_k):
    """Candidate controller chunks [N,H,7]; score aligned prefix vs previous leftover."""
    if prev_leftover is None:
        return None
    horizon = candidate_actions.shape[1] if candidate_actions.ndim == 3 else len(candidate_actions)
    # candidate_actions may be list; normalized below
    return None


def run_episode(env, clean_env, policy, language, initial_state, method, source, args, device, seed, style_stats=None):
    # Mirror evaluate_libero.py's per-episode seeding.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    initial_image_mean = float(np.asarray(obs["agentview_image"]).mean())
    done = False
    success = False
    steps = 0
    for _ in range(SETTLING_STEPS):
        obs, reward, done, info = env.step(DUMMY_ACTION)
        success = is_success(env, info, reward, done)
        if success or done:
            break
    estimator = None
    if method != "frozen":
        estimator = TargetOnlyMeanEstimator(
            source, momentum=0.5, alpha=args.alpha, clip_norm=args.clip_norm)
    chunk_index = 0
    held_correction = None
    executed_actions = []
    correction_norms = []
    correction_stats = []
    paired_gap_stats = []
    preclip_saturation = []
    update_count = 0
    application_count = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    prev_leftover = None
    selection_records = []
    while not success and not done and steps < args.max_steps:
        policy_obs = policy_observation(obs, language, policy)
        if method == "frozen_predict":
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(args.precision == "bf16" and device.type == "cuda")):
                controller = policy.predict(policy_obs)[0].float().cpu().numpy()
            for k in range(args.executed_k):
                raw_action = controller[k]
                preclip_saturation.append(float(np.any(np.abs(raw_action) > 1.0)))
                clipped = np.clip(raw_action, -1.0, 1.0)
                executed_actions.append(clipped.astype(np.float32))
                obs, reward, done, info = env.step(clipped)
                steps += 1
                success = is_success(env, info, reward, done)
                if success or done or steps >= args.max_steps:
                    break
            chunk_index += 1
            continue
        if method == "leftover_select":
            if style_stats is None:
                raise RuntimeError("leftover_select requires source style statistics")
            visual_raw, base_rep, patches = visual_from_obs(policy, policy_obs, return_patches=True)
            source_mean, _ = style_stats
            mean_patches = {v: style_mean_only(patches[v], source_mean[v], args.style_strength)
                            for v in patches}
            _, pooled_lang = policy.representation_extractor.language_encoder([language])
            visual_mean = policy.representation_extractor.cross_attention_resampler(
                mean_patches, pooled_lang)
            z0 = torch.randn((1, policy.action_generator.action_horizon,
                              policy.action_generator.action_dim),
                             device=device, dtype=torch.float32, generator=generator)
            candidates = []
            for visual_candidate in (visual_raw, visual_mean):
                rep_candidate = make_rep(visual_candidate, base_rep)
                norm = sample_z(policy, rep_candidate, z0)
                ctrl = policy.normalizer.unnormalize_action(norm)[0].float().cpu().numpy()
                candidates.append(ctrl)
            if prev_leftover is None:
                chosen = 0
                scores = []
            else:
                L = prev_leftover.shape[0]
                scores = [float(np.mean((c[:L] - prev_leftover) ** 2)) for c in candidates]
                chosen = int(np.argmin(scores))
            controller = candidates[chosen]
            for k in range(args.executed_k):
                raw_action = controller[k]
                preclip_saturation.append(float(np.any(np.abs(raw_action) > 1.0)))
                clipped = np.clip(raw_action, -1.0, 1.0)
                executed_actions.append(clipped.astype(np.float32))
                obs, reward, done, info = env.step(clipped)
                steps += 1
                success = is_success(env, info, reward, done)
                if success or done or steps >= args.max_steps:
                    break
            prev_leftover = controller[args.executed_k:].copy()
            selection_records.append({"chunk": chunk_index, "chosen": chosen,
                                      "scores": scores})
            chunk_index += 1
            continue
        if method not in ("style_norm", "style_mean"):
            visual, base_rep = visual_from_obs(policy, policy_obs)
        context = policy_obs["state"].detach().cpu().numpy()
        correction = None
        style_used = False
        paired_gap_norm = None
        shift_clean_image_diff = None
        if method in ("style_norm", "style_mean"):
            if style_stats is None:
                raise RuntimeError("style_norm requires source style statistics")
            visual_raw, base_rep, patches = visual_from_obs(policy, policy_obs, return_patches=True)
            source_mean, source_std = style_stats
            if method == "style_norm":
                norm_patches = {v: style_transfer(patches[v], source_mean[v], source_std[v])
                                for v in patches}
            else:
                norm_patches = {v: style_mean_only(patches[v], source_mean[v],
                                                   args.style_strength)
                                for v in patches}
            _, pooled_lang = policy.representation_extractor.language_encoder([language])
            visual = policy.representation_extractor.cross_attention_resampler(
                norm_patches, pooled_lang)
            rep = SimpleNamespace(
                tokens=torch.cat([visual, base_rep.tokens[:, visual.shape[1]:]], dim=1),
                pooled_language=base_rep.pooled_language,
                action_dtype=visual.dtype,
            )
            visual_used = visual
            style_used = True
        elif method == "paired_first" and chunk_index == 0:
            if clean_env is None:
                raise RuntimeError("paired_first requires a clean twin environment")
            clean_obs = render_policy_observation(clean_env, snapshot(env))
            shift_clean_image_diff = float(np.abs(
                np.asarray(obs["agentview_image"], dtype=np.float32)
                - np.asarray(clean_obs["agentview_image"], dtype=np.float32)).mean())
            clean_policy_obs = policy_observation(clean_obs, language, policy)
            clean_visual, _ = visual_from_obs(policy, clean_policy_obs)
            # The method is a global mean correction; compute the paired gap
            # on pooled tokens, not per-token, so it matches estimator space.
            paired_gap = visual.mean(dim=1) - clean_visual.mean(dim=1)
            paired_gap = paired_gap.unsqueeze(1)
            paired_gap_norm = float(paired_gap.norm())
            # Use the same amplitude rule as the estimator: L2 clip then alpha.
            paired_scale = min(1.0, float(args.clip_norm) / (paired_gap_norm + 1e-12))
            held_correction = float(args.alpha) * paired_scale * paired_gap
            correction = held_correction
            paired_gap_stats.append({
                "gap_norm": paired_gap_norm,
                "preclip_correction_norm": paired_gap_norm,
                "applied_correction_norm": float(held_correction.norm()),
                "clip_scale": paired_scale,
                "shift_clean_image_mean_abs_diff": shift_clean_image_diff,
            })
        if estimator is not None:
            pooled = visual.mean(dim=1)
            if method == "first_observation" and chunk_index == 0:
                estimator.update(pooled, context)
                update_count += 1
                held_correction = estimator.applied_correction()
                correction = held_correction
            elif method == "online":
                estimator.update(pooled, context)
                update_count += 1
                correction = estimator.applied_correction()
            elif held_correction is not None:
                correction = held_correction
        if correction is not None:
            applied = float(correction.norm())
            ema_norm = float(estimator.ema.norm()) if estimator.ema is not None else 0.0
            raw_norm = (float(estimator.last_raw_residual.norm())
                        if getattr(estimator, "last_raw_residual", None) is not None else 0.0)
            preclip = float(estimator.alpha) * ema_norm
            correction_norms.append(applied)
            correction_stats.append({
                "raw_residual_norm": raw_norm,
                "ema_norm": ema_norm,
                "preclip_correction_norm": preclip,
                "applied_correction_norm": applied,
                "clip_scale": applied / (preclip + 1e-12),
            })
            application_count += 1
        if not style_used:
            visual_used = visual if correction is None else visual - correction
        z0 = torch.randn((1, policy.action_generator.action_horizon,
                          policy.action_generator.action_dim),
                         device=device, dtype=torch.float32, generator=generator)
        action_norm = sample_z(policy, make_rep(visual_used, base_rep), z0)
        controller = policy.normalizer.unnormalize_action(action_norm)[0].float().cpu().numpy()
        for k in range(args.executed_k):
            raw_action = controller[k]
            preclip_saturation.append(float(np.any(np.abs(raw_action) > 1.0)))
            clipped = np.clip(raw_action, -1.0, 1.0)
            executed_actions.append(clipped.astype(np.float32))
            obs, reward, done, info = env.step(clipped)
            steps += 1
            success = is_success(env, info, reward, done)
            if success or done or steps >= args.max_steps:
                break
        chunk_index += 1
    import hashlib
    arr = np.stack(executed_actions) if executed_actions else np.zeros((0, 7), np.float32)
    return {"method": method, "init_index": None,
            "initial_agentview_mean": initial_image_mean,
            "paired_first": method == "paired_first",
            "success": bool(success),
            "steps": int(steps), "chunks": int(chunk_index),
            "action_abs_mean": float(np.abs(arr).mean()) if len(arr) else None,
            "action_saturation_fraction": float((np.abs(arr) >= 1.0).mean()) if len(arr) else None,
            "preclip_action_saturation_fraction": (float(np.mean(preclip_saturation))
                                                    if preclip_saturation else None),
            "correction_norms": correction_norms,
            "correction_stats": correction_stats,
            "paired_gap_stats": paired_gap_stats,
            "update_count": update_count,
            "application_count": application_count,
            "selection_records": selection_records,
            "action_sha256": hashlib.sha256(arr.tobytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--init-indices", default="3,5")
    ap.add_argument("--domain", choices=["shift", "clean"], default="shift")
    ap.add_argument("--tta-mode", choices=["episodic"], default="episodic",
                    help="episodic TTA: fresh adapter/estimator per episode")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--lighting-id", type=int, default=2222)
    ap.add_argument("--methods", default="frozen,first_observation,online")
    ap.add_argument("--max-steps", type=int, default=280)
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--style-strength", type=float, default=1.0)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--clip-norm", type=float, default=20.0)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--content-rank", type=int, default=2)
    ap.add_argument("--source-folds", type=int, default=5)
    ap.add_argument("--source-fit-start", type=int, default=10)
    ap.add_argument("--source-fit-count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--seed-offsets", default="0")
    ap.add_argument("--allow-non-heldout", action="store_true",
                    help="protocol audit only; permits dev states")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == args.suite)
    variant = next(v for v in task["variants"]["lighting"]
                   if v["classification_id"] == args.lighting_id)
    base_bddl, shift_bddl = variant["base_bddl"], variant["env_bddl_name"]
    env_bddl = base_bddl if args.domain == "clean" else shift_bddl
    language = BDDLUtils.get_problem_info(base_bddl)["language_instruction"]
    methods = validate_methods(args.methods)
    needs_source = any(m not in {"frozen", "frozen_predict", "paired_first"} for m in methods)
    fit_ids = []
    source = None
    diag = {}
    style_mean = None
    style_std = None
    if needs_source:
        manifest = json.loads((args.capture / "manifest.json").read_text())
        z = np.load(args.capture / "features.npz")
        rows = [r for r in manifest["records"] if r.get("compare_pass")]
        by_state = {int(r["index"]): int(r["feature_row"]) for r in rows}
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    if needs_source:
        fit_ids, source, diag = build_source(args, policy, manifest, z, by_state, device, language)
        fit_ids_style, style_mean, style_std = build_style_stats(
            args, policy, manifest, z, by_state, device, language)
        requested = args.source_fit_count
        if len(fit_ids) != requested:
            raise ValueError(f"source fit returned {len(fit_ids)} states, requested {requested}")
    initial_states = torch.load(task["state_split"]["source_file"], weights_only=False)
    init_indices = parse_init_indices(args.init_indices)
    horizon = int(policy.action_generator.action_horizon)
    validate_executed_k(args.executed_k, horizon)
    heldout = set(int(i) for i in task["state_split"]["heldout_indices"])
    validate_init_indices(init_indices, len(initial_states), heldout, fit_ids,
                          require_heldout=not args.allow_non_heldout)
    records = []
    seed_offsets = [int(x) for x in args.seed_offsets.split(",")]
    for init_index in init_indices:
        for method in methods:
            for seed_offset in seed_offsets:
                seed = args.seed + init_index + seed_offset
                env = OffScreenRenderEnv(bddl_file_name=env_bddl, camera_heights=256,
                                         camera_widths=256, render_gpu_device_id=args.render_gpu)
                clean_env = None
                if method == "paired_first":
                    clean_env = OffScreenRenderEnv(bddl_file_name=base_bddl, camera_heights=256,
                                                   camera_widths=256, render_gpu_device_id=args.render_gpu)
                try:
                    rec = run_episode(env, clean_env, policy, language, initial_states[init_index],
                                      method, source, args, device, seed,
                                      (style_mean, style_std)
                                      if method in ("style_norm", "style_mean", "leftover_select")
                                      else None)
                    rec["init_index"] = init_index
                    rec["seed_offset"] = seed_offset
                    records.append(rec)
                    print(json.dumps(rec), flush=True)
                finally:
                    env.close()
                    if clean_env is not None:
                        clean_env.close()
    payload = {"schema": "closed_loop_tta_pilot_v2", "fit_ids": fit_ids, "domain": args.domain,
               "script_sha256": sha256_file(Path(__file__)),
               "checkpoint_sha256": sha256_file(args.checkpoint),
               "capture_manifest_sha256": sha256_file(args.capture / "manifest.json"),
               "capture_features_sha256": sha256_file(args.capture / "features.npz"),
               "matrix_sha256": sha256_file(args.matrix),
               "libero_source": str(args.libero_src),
               "provenance_complete": all([
                   args.checkpoint.exists(), (args.capture / "manifest.json").exists(),
                   (args.capture / "features.npz").exists(), args.matrix.exists()]),
               "source_diagnostics": diag, "records": records,
               "methods": methods, "init_indices": init_indices,
               "claim_scope": "exploratory pilot; not statistical significance",
               "protocol": {"tta_mode": args.tta_mode,
                            "episode_reset": True,
                            "settling_steps": SETTLING_STEPS, "max_steps": args.max_steps,
                            "executed_k": args.executed_k, "action_clip": [-1, 1],
                            "success": "official is_success"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
