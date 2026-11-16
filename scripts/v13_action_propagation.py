"""V1.3 pilot: action propagation with Oracle/Random gates and common noise.

Strict split: Gate dimensions come from Dev s_pair only; held-out states are
reported separately and are never used to choose dimensions.  Real 8D policy
state is used.  No simulator action or rollout.
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


@torch.no_grad()
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


def _unnorm(a, normalizer):
    return normalizer.unnormalize_action(a)


def _diff(a, b, normalizer):
    a = _unnorm(a, normalizer); b = _unnorm(b, normalizer)
    d = (a - b).abs()
    scale = (normalizer.action_max - normalizer.action_min).to(d.device).clamp_min(1e-6)
    return {"xyz": float(d[..., :3].mean().cpu()), "rot": float(d[..., 3:6].mean().cpu()),
            "gripper": float(d[..., 6].mean().cpu()), "all": float(d.mean().cpu()),
            "norm_all": float((d / scale).mean().cpu())}


def action_metrics(a_c, a_s, a_g, normalizer):
    dc = _unnorm(a_c, normalizer); ds = _unnorm(a_s, normalizer); dg = _unnorm(a_g, normalizer)
    v_cs = (dc - ds).reshape(-1); v_gs = (dg - ds).reshape(-1)
    denom = float(v_cs.norm() * v_gs.norm() + 1e-12)
    return {
        "pair": _diff(a_s, a_c, normalizer),
        "gate_oracle": _diff(a_g, a_s, normalizer),
        "gate_to_clean": _diff(a_g, a_c, normalizer),
        "align_cos": float(torch.dot(v_gs, v_cs).cpu() / denom),
        "align_proj": float(torch.dot(v_gs, v_cs).cpu() / (torch.dot(v_cs, v_cs).cpu() + 1e-12)),
    }


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
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--random-gates", type=int, default=8)
    ap.add_argument("--z0-count", type=int, default=10)
    ap.add_argument("--noise-pairs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    configure(args)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    manifest = json.loads((args.dir / "manifest.json").read_text())
    scores = json.loads((args.dir / "scores_per_view.json").read_text())
    z = np.load(args.dir / "features.npz")
    valid = [r["index"] for r in manifest["records"] if r.get("compare_pass")]
    pos = {s: j for j, s in enumerate(valid)}
    dev_idx = [i for i in manifest["dev_indices"] if i in pos]
    held_idx = [i for i in manifest["heldout_indices"] if i in pos]
    views = manifest["policy_views"]
    policy = load_policy(CKPT, device=args.device); policy.eval()
    clean_states = torch.load(args.clean_init_file, weights_only=False)

    # real 8D state per valid state, from the clean env
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

    oracle_dims = np.argsort(-np.asarray(scores["views"][args.target_view]["pair_dev"]))[:args.k].tolist()
    # Fixed random masks, pre-selected on Dev energy only; reused for all states.
    dev_pos = [pos[i] for i in dev_idx]
    dev_feat = z[f"shift_{args.target_view}"][dev_pos].astype(np.float64)
    dim_energy = (dev_feat ** 2).mean(axis=(0, 1))
    oracle_energy_target = float(dim_energy[oracle_dims].sum())
    rng0 = np.random.RandomState(args.seed)
    candidates = [rng0.choice(384, size=args.k, replace=False) for _ in range(500)]
    cand_energy = [float(dim_energy[c].sum()) for c in candidates]
    order = np.argsort(np.abs(np.asarray(cand_energy) - oracle_energy_target))
    random_gates = [candidates[j].tolist() for j in order[:args.random_gates]]
    device = next(policy.parameters()).device
    H, A = policy.action_generator.action_horizon, policy.action_generator.action_dim
    out = {"num_states": len(valid), "dev": len(dev_idx), "heldout": len(held_idx),
           "target_view": args.target_view, "k": args.k, "oracle_dims": oracle_dims,
           "random_gates_n": args.random_gates, "random_gates": random_gates,
           "energy_match": {"dev_oracle_target": oracle_energy_target,
                            "selected_candidate_energies": [cand_energy[j] for j in order[:args.random_gates]]},
           "z0_count": args.z0_count,
           "noise_pairs": args.noise_pairs, "seed": args.seed, "precision": "fp32",
           "per_state": []}
    for state in valid:
        j = pos[state]
        feats = {v: z[f"shift_{v}"][j].copy() for v in views}
        clean_feats = {v: z[f"clean_{v}"][j].copy() for v in views}
        state8 = states8[state]
        rep_c = repr_from_features(policy, clean_feats, state8, args.clean_language)
        rep_s = repr_from_features(policy, feats, state8, args.clean_language)
        rep_go = repr_from_features(policy, gate_features(feats, args.target_view, oracle_dims),
                                    state8, args.clean_language)
        rep_gr = [repr_from_features(policy, gate_features(feats, args.target_view, dims),
                                     state8, args.clean_language) for dims in random_gates]
        oracle_energy = float(np.square(feats[args.target_view][..., oracle_dims]).sum())
        random_energy = float(np.mean([np.square(feats[args.target_view][..., d]).sum()
                                       for d in random_gates]))
        torch.manual_seed(args.seed + state)
        z0s = [torch.randn(1, H, A, device=device) for _ in range(args.z0_count)]
        zs = [torch.randn(1, H, A, device=device) for _ in range(args.noise_pairs)]
        per = []
        for z0 in z0s:
            a_c = sample_with_z0(policy.action_generator, rep_c, z0)
            a_s = sample_with_z0(policy.action_generator, rep_s, z0)
            a_go = sample_with_z0(policy.action_generator, rep_go, z0)
            m = action_metrics(a_c, a_s, a_go, policy.normalizer)
            gr = [action_metrics(a_c, a_s, sample_with_z0(policy.action_generator, r, z0),
                                 policy.normalizer) for r in rep_gr]
            m["gate_random"] = {key: {sub: float(np.mean([x["gate_oracle"][sub] for x in gr]))
                                      for sub in m["gate_oracle"]}
                                for key in ("gate_random",)}["gate_random"]
            m["gate_random_to_clean"] = {sub: float(np.mean([x["gate_to_clean"][sub] for x in gr]))
                                         for sub in m["gate_to_clean"]}
            m["gate_random_align_cos"] = float(np.mean([x["align_cos"] for x in gr]))
            m["gate_random_align_proj"] = float(np.mean([x["align_proj"] for x in gr]))
            per.append(m)
        noise = []
        for i in range(args.noise_pairs):
            z1, z2 = zs[i], zs[(i + 1) % len(zs)]
            noise.append(_diff(sample_with_z0(policy.action_generator, rep_s, z1),
                               sample_with_z0(policy.action_generator, rep_s, z2),
                               policy.normalizer))
        def avg(dcts):
            out = {}
            for k, first in dcts[0].items():
                if isinstance(first, dict):
                    out[k] = {m: float(np.mean([d[k][m] for d in dcts])) for m in first}
                else:
                    out[k] = float(np.mean([d[k] for d in dcts]))
            return out
        rec = {"state": state, "split": "dev" if state in dev_idx else "heldout",
               "oracle_energy": oracle_energy, "random_energy": random_energy,
               "aggregate": avg(per),
               "noise": {m: float(np.mean([d[m] for d in noise])) for m in noise[0]}}
        out["per_state"].append(rec)
    def agg(records):
        out = {}
        for k, first in records[0]["aggregate"].items():
            if isinstance(first, dict):
                out[k] = {m: float(np.mean([r["aggregate"][k][m] for r in records])) for m in first}
            else:
                out[k] = float(np.mean([r["aggregate"][k] for r in records]))
        return out
    for split, ids in (("dev", dev_idx), ("heldout", held_idx)):
        rs = [r for r in out["per_state"] if r["state"] in ids]
        out[split] = {"aggregate": agg(rs),
                      "noise": {m: float(np.mean([r["noise"][m] for r in rs])) for m in rs[0]["noise"]},
                      "oracle_energy": float(np.mean([r["oracle_energy"] for r in rs])),
                      "random_energy": float(np.mean([r["random_energy"] for r in rs]))}
    (args.dir / "v13_action_propagation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"dev": out["dev"], "heldout": out["heldout"]}, indent=2))


if __name__ == "__main__":
    main()
