"""V1.3 pilot: does the Camera-sensitive feature change propagate to actions?

Uses saved V1.2 features/images, real 8D policy state from the clean env, and a
common initial action noise z0.  Compares:
  A_C(z0), A_S(z0), A_G_oracle(z0), A_G_random(z0)
  A_S(z1) vs A_S(z2) as sampling-noise floor.
No simulator action, no rollout, no success labels.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from corp import load_policy  # noqa: E402
from corp.representation.extractor import Representation  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, state_from_observation  # noqa: E402

K_DEFAULT = 20


def sample_with_z0(generator, representation, z0, num_steps=None):
    steps = generator.num_sample_steps if num_steps is None else num_steps
    action = z0.to(device=representation.tokens.device, dtype=representation.action_dtype).clone()
    prefix = generator.prepare_prefix(representation)
    for step in range(steps):
        tau = torch.full((action.shape[0],), step / float(steps),
                         device=action.device, dtype=action.dtype)
        action = action + generator(action, tau, prefix) * (1.0 / float(steps))
    return action


def repr_from_features(policy, features_by_view, state8, instruction):
    ext = policy.representation_extractor
    device = next(policy.parameters()).device
    patch = {v: torch.as_tensor(f, dtype=torch.float32, device=device).unsqueeze(0)
             for v, f in features_by_view.items()}
    lang_tokens, pooled = ext.language_encoder(instruction)
    visual = ext.cross_attention_resampler(patch, pooled)
    st = torch.as_tensor(state8, dtype=torch.float32, device=device).unsqueeze(0)
    st = policy.normalizer.normalize_state(st)
    state_token = ext.state_encoder(st)
    tokens = torch.cat([visual, lang_tokens.to(visual.dtype), state_token], dim=1)
    return Representation(tokens=tokens, pooled_language=pooled, action_dtype=visual.dtype)


def gate_features(features_by_view, target_view, dims):
    out = {v: f.copy() for v, f in features_by_view.items()}
    out[target_view][..., dims] = 0.0
    return out


def action_diff(a, b, normalizer):
    a = normalizer.unnormalize_action(a); b = normalizer.unnormalize_action(b)
    d = (a - b).abs()
    scale = (normalizer.action_max - normalizer.action_min).to(d.device).clamp_min(1e-6)
    groups = {"xyz": d[..., :3].mean(), "rot": d[..., 3:6].mean(), "gripper": d[..., 6].mean(),
              "all": d.mean(), "norm_all": (d / scale).mean()}
    return {k: float(v.cpu()) for k, v in groups.items()}


def configure(args):
    import os, yaml
    src = args.libero_src.expanduser().resolve(); root = src / "libero" / "libero"
    sys.path.insert(0, str(src))
    cfg = args.libero_config.expanduser().resolve(); cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "datasets").mkdir(exist_ok=True)
    (cfg / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"), "assets": str(root / "assets"),
        "datasets": str(cfg / "datasets")}))
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    for name, value in {"int": int, "float": float, "bool": bool, "str": str}.items():
        if name not in np.__dict__: setattr(np, name, value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--clean-bddl", type=Path, required=True)
    ap.add_argument("--clean-init-file", type=Path, required=True)
    ap.add_argument("--clean-language", required=True)
    ap.add_argument("--clean-suite", default="libero_spatial")
    ap.add_argument("--target-view", default="agentview")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--random-gates", type=int, default=20)
    ap.add_argument("--noise-pairs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    configure(args)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    manifest = json.loads((args.dir / "manifest.json").read_text())
    scores = json.loads((args.dir / "scores_per_view.json").read_text())
    z = np.load(args.dir / "features.npz")
    valid = [r["index"] for r in manifest["records"] if r.get("compare_pass")]
    valid_pos = {s: j for j, s in enumerate(valid)}
    views = manifest["policy_views"]
    assert args.target_view in views
    policy = load_policy(CKPT, device=args.device); policy.eval()
    clean_states = torch.load(args.clean_init_file, weights_only=False)

    suite = benchmark.get_benchmark_dict()[args.clean_suite]()
    env = OffScreenRenderEnv(bddl_file_name=str(args.clean_bddl), camera_heights=256,
                             camera_widths=256, render_gpu_device_id=0)
    states8 = {}
    try:
        for state in valid:
            env.seed(0); env.reset(); env.set_init_state(clean_states[state])
            obs = None
            for _ in range(SETTLING_STEPS):
                obs, _, done, _ = env.step(DUMMY_ACTION)
                if done: raise RuntimeError(f"settling done at state {state}")
            states8[state] = state_from_observation(obs, policy.normalizer).astype(np.float32)
    finally:
        env.close()
    rng = np.random.RandomState(args.seed)
    dev_idx = [i for i in manifest["dev_indices"] if i in valid]
    oracle_dims = np.argsort(-np.asarray(scores["views"][args.target_view]["pair_dev"]))[:args.k].tolist()
    random_gates = [rng.choice(384, size=args.k, replace=False).tolist() for _ in range(args.random_gates)]
    device = next(policy.parameters()).device
    H, A = policy.action_generator.action_horizon, policy.action_generator.action_dim
    out = {"num_states": len(valid), "target_view": args.target_view, "k": args.k,
           "oracle_dims": oracle_dims, "random_gates": random_gates,
           "noise_pairs": args.noise_pairs, "per_state": []}
    aggregate = {k: [] for k in ("pair", "gate_oracle", "gate_random", "noise")}
    for state in valid:
        j = valid_pos[state]
        feats = {v: z[f"shift_{v}"][j].copy() for v in views}
        clean_feats = {v: z[f"clean_{v}"][j].copy() for v in views}
        state8 = states8[state]
        rep_c = repr_from_features(policy, clean_feats, state8, args.clean_language)
        rep_s = repr_from_features(policy, feats, state8, args.clean_language)
        gate_oracle = gate_features(feats, args.target_view, oracle_dims)
        rep_go = repr_from_features(policy, gate_oracle, state8, args.clean_language)
        z0 = torch.randn(1, H, A, device=device)
        a_c = sample_with_z0(policy.action_generator, rep_c, z0)
        a_s = sample_with_z0(policy.action_generator, rep_s, z0)
        a_go = sample_with_z0(policy.action_generator, rep_go, z0)
        pair = action_diff(a_s, a_c, policy.normalizer)
        gate_o = action_diff(a_go, a_s, policy.normalizer)
        # random gates on the same Shift features
        rand_diffs = []
        for dims in random_gates:
            r = repr_from_features(policy, gate_features(feats, args.target_view, dims),
                                   state8, args.clean_language)
            rand_diffs.append(action_diff(sample_with_z0(policy.action_generator, r, z0), a_s,
                                          policy.normalizer))
        gate_r = {k: float(np.mean([r[k] for r in rand_diffs])) for k in pair}
        # sampling noise: same Shift features, different z
        noise_diffs = []
        for _ in range(args.noise_pairs):
            z1 = torch.randn(1, H, A, device=device); z2 = torch.randn(1, H, A, device=device)
            noise_diffs.append(action_diff(sample_with_z0(policy.action_generator, rep_s, z1),
                                           sample_with_z0(policy.action_generator, rep_s, z2),
                                           policy.normalizer))
        noise = {k: float(np.mean([d[k] for d in noise_diffs])) for k in pair}
        for key, val in (("pair", pair), ("gate_oracle", gate_o), ("gate_random", gate_r), ("noise", noise)):
            aggregate[key].append(val)
        out["per_state"].append({"state": state, "pair": pair, "gate_oracle": gate_o,
                                 "gate_random": gate_r, "noise": noise})
    out["aggregate"] = {k: {m: float(np.mean([d[m] for d in v])) for m in v[0]}
                        for k, v in aggregate.items()}
    (args.dir / "v13_action_propagation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["aggregate"], indent=2))


if __name__ == "__main__":
    main()
