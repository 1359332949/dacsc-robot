"""Action-distance validation for source-calibrated Target-only Mean.

4 held-out target trajectories, episode-independent EMA, same z0 per state.
Clean observations are used only for offline reference/Oracle actions.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS  # noqa: E402
from dacsc.state_pairing import snapshot, restore  # noqa: E402
from dacsc.observation_render import render_policy_observation  # noqa: E402
from dacsc.target_only_mean import ConditionalSourceBaseline, TargetOnlyMeanEstimator  # noqa: E402


def configure(args):
    import os, yaml
    src = args.libero_src.expanduser().resolve(); root = src / "libero" / "libero"
    sys.path.insert(0, str(src))
    cfg = args.libero_config.expanduser().resolve(); cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "datasets").mkdir(exist_ok=True)
    cp = cfg / "config.yaml"
    if not cp.is_file():
        cp.write_text(yaml.safe_dump({"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
                                      "init_states": str(root / "init_files"), "assets": str(root / "assets"),
                                      "datasets": str(cfg / "datasets")}))
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def normalize_action(policy, a):
    return policy.normalizer.normalize_action(a)


@torch.no_grad()
def sample_z(policy, representation, z0):
    gen = policy.action_generator
    action = z0.to(device=representation.tokens.device, dtype=representation.action_dtype).clone()
    prefix = gen.prepare_prefix(representation)
    for step in range(gen.num_sample_steps):
        tau = torch.full((action.shape[0],), step / float(gen.num_sample_steps),
                         device=action.device, dtype=action.dtype)
        action = action + gen(action, tau, prefix) * (1.0 / float(gen.num_sample_steps))
    return action


@torch.no_grad()
def visual_from_obs(policy, obs, instruction):
    batch = prepare_observation(obs, config=policy.config, normalizer=policy.normalizer,
                                device=next(policy.parameters()).device)
    ext = policy.representation_extractor
    views = tuple(ext.view_names)
    patches = ext.visual_encoder(torch.cat([batch["images"][v] for v in views], dim=0))
    b = batch["images"][views[0]].shape[0]
    patch = {v: patches[i*b:(i+1)*b] for i, v in enumerate(views)}
    lang, pooled = ext.language_encoder(batch["instruction"])
    visual = ext.cross_attention_resampler(patch, pooled)
    state = ext.state_encoder(batch["state"])
    rep = __import__("types").SimpleNamespace(
        tokens=torch.cat([visual, lang.to(visual.dtype), state], dim=1),
        pooled_language=pooled, action_dtype=visual.dtype)
    return visual, rep, batch["state"]


def action_metrics(policy, pred, clean):
    p = policy.normalizer.unnormalize_action(pred); c = policy.normalizer.unnormalize_action(clean)
    d = (p - c).abs()
    return {"normalized_mae": float(d.mean()), "xyz_mae": float(d[..., :3].mean()),
            "rot_mae": float(d[..., 3:6].mean()), "gripper_mae": float(d[..., 6].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trajectories", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--clip-norm", type=float, default=20.0)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == "libero_object")
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == 2222)
    base_bddl = variant["base_bddl"]; shift_bddl = variant["env_bddl_name"]
    language = BDDLUtils.get_problem_info(base_bddl)["language_instruction"]
    policy = load_policy(args.checkpoint, device="cuda"); policy.eval()
    # Fit source baseline and pooled paired mean on development fit states.
    manifest = json.loads((args.capture / "manifest.json").read_text())
    z = np.load(args.capture / "features.npz")
    rows = [r for r in manifest["records"] if r.get("compare_pass")]
    by = {int(r["index"]): int(r["feature_row"]) for r in rows}
    dev = [i for i in manifest["dev_indices"] if i in by]
    fit_ids = dev[10:20]
    def pooled(state, kind):
        v = {view: z[f"{kind}_{view}"][by[state]] for view in manifest["policy_views"]}
        # use the same DINO/resampler path directly
        ext = policy.representation_extractor
        p = {k: torch.from_numpy(v[k]).float().unsqueeze(0).cuda()
             for k in manifest["policy_views"]}
        lang, pl = ext.language_encoder(language)
        visual = ext.cross_attention_resampler(p, pl)
        return visual.mean(dim=1)[0].detach().cpu().numpy()
    clean_fit = {s: pooled(s, "clean") for s in fit_ids}
    shift_fit = {s: pooled(s, "shift") for s in fit_ids}
    X = np.stack([z["state8"][by[s]] for s in fit_ids])
    source = ConditionalSourceBaseline(l2=args.l2); source.fit(X, np.stack([clean_fit[s] for s in fit_ids]))
    paired = np.mean([shift_fit[s] - clean_fit[s] for s in fit_ids], axis=0)
    held = task["state_split"]["heldout_indices"][:args.trajectories]
    init = torch.load(task["state_split"]["source_file"], weights_only=False)
    env_s = OffScreenRenderEnv(bddl_file_name=shift_bddl, camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    env_c = OffScreenRenderEnv(bddl_file_name=base_bddl, camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    records = []
    try:
        for idx in held:
            env_s.seed(0); # shift env not reset: preserve Lighting assets
            obs = env_s.set_init_state(init[int(idx)])
            for _ in range(SETTLING_STEPS):
                obs, _, done, _ = env_s.step(DUMMY_ACTION)
            estimator = TargetOnlyMeanEstimator(source, momentum=0.5, alpha=args.alpha,
                                                clip_norm=args.clip_norm)
            for c in range(args.chunks):
                snap = snapshot(env_s)
                shift_obs = {"images": {}, "state": None, "instruction": [language]}
                # render policy observation from the same state without advancing
                from evaluate_libero import policy_observation
                po_s = policy_observation(obs, language, policy)
                vs, rep_s, _ = visual_from_obs(policy, po_s, language)
                # clean twin rendering from same physical state
                clean_obs = render_policy_observation(env_c, snap)
                po_c = policy_observation(clean_obs, language, policy)
                vc, rep_c, _ = visual_from_obs(policy, po_c, language)
                # Target-only estimate uses only shift history.  The estimator
                # is reset once per episode and updated at each decision point.
                st8 = policy_observation(obs, language, policy)["state"].numpy()
                est = estimator.update(vs.mean(dim=1), st8)
                corr_target = torch.as_tensor(est, device="cuda", dtype=vs.dtype).view(1, 1, -1)
                correction_paired = torch.as_tensor(paired, device="cuda", dtype=vs.dtype).view(1, 1, -1)
                z0 = torch.randn(1, policy.action_generator.action_horizon,
                                 policy.action_generator.action_dim, device="cuda")
                def act(visual):
                    rep = __import__("types").SimpleNamespace(
                        tokens=torch.cat([visual, rep_s.tokens[:, visual.shape[1]:]], dim=1),
                        pooled_language=rep_s.pooled_language, action_dtype=visual.dtype)
                    return sample_z(policy, rep, z0)
                a_clean = act(vc)
                a_frozen = act(vs)
                a_paired = act(vs - correction_paired)
                a_target = act(vs - corr_target)
                a_oracle = a_clean
                rec = {"init_index": int(idx), "chunk": c,
                       "clean_vs_oracle": action_metrics(policy, a_clean, a_clean),
                       "frozen": action_metrics(policy, a_frozen, a_clean),
                       "paired_fit_mean": action_metrics(policy, a_paired, a_clean),
                       "target_only_mean": action_metrics(policy, a_target, a_clean),
                       "oracle": action_metrics(policy, a_oracle, a_clean),
                       "target_residual_norm": float(np.linalg.norm(est.detach().cpu().numpy())),
                       "clean_residual_norm": float(np.linalg.norm(vc.mean(1)[0].detach().cpu().numpy() - source.predict(st8[None, :])[0]))}
                records.append(rec)
                actions = a_frozen[0].detach().cpu().numpy()
                for a in actions[:args.executed_k]:
                    obs, reward, done, info = env_s.step(np.clip(a, -1, 1))
                    if done: break
    finally:
        env_s.close(); env_c.close()
    summary = {}
    for k in ("frozen", "paired_fit_mean", "target_only_mean", "oracle"):
        summary[k] = {m: float(np.mean([r[k][m] for r in records])) for m in ("normalized_mae", "xyz_mae", "rot_mae", "gripper_mae")}
    out = {"records": records, "summary": summary, "paired_norm": float(np.linalg.norm(paired)),
           "capture_manifest_sha256": __import__("hashlib").sha256((args.capture/"manifest.json").read_bytes()).hexdigest()}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
