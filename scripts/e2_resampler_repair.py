"""E2 pilot: Oracle paired repair at the visual Resampler output.

Uses captured DINO patch features from v12.  For the same physical state:
  V_c = Resampler(F_c), V_s = Resampler(F_s)
Repairs are applied to V_s and fed to the frozen action generator.
All repairs using V_c are Oracle diagnostics, not deployable TTA.
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
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, state_from_observation  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402


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


@torch.no_grad()
def sample_with_z0(generator, representation, z0, num_steps=None):
    steps = generator.num_sample_steps if num_steps is None else num_steps
    action = z0.to(device=representation.tokens.device, dtype=representation.action_dtype).clone()
    prefix = generator.prepare_prefix(representation)
    for step in range(steps):
        tau = torch.full((action.shape[0],), step / float(steps), device=action.device, dtype=action.dtype)
        action = action + generator(action, tau, prefix) * (1.0 / float(steps))
    return action


def visual_tokens(policy, patch_by_view, instruction):
    ext = policy.representation_extractor
    device = next(policy.parameters()).device
    patches = {v: torch.as_tensor(f, dtype=torch.float32, device=device).unsqueeze(0)
               for v, f in patch_by_view.items()}
    lang_tokens, pooled = ext.language_encoder(instruction)
    return ext.cross_attention_resampler(patches, pooled), lang_tokens, pooled


def repr_from_visual(policy, visual, lang_tokens, pooled, state8):
    device = next(policy.parameters()).device
    st = torch.as_tensor(state8, dtype=torch.float32, device=device).unsqueeze(0)
    st = policy.normalizer.normalize_state(st)
    state_token = policy.representation_extractor.state_encoder(st)
    tokens = torch.cat([visual, lang_tokens.to(visual.dtype), state_token], dim=1)
    return Representation(tokens=tokens, pooled_language=pooled, action_dtype=visual.dtype)


def action_metrics(a_c, a_s, a_m, normalizer):
    def unnorm(a): return normalizer.unnormalize_action(a)
    c, s, m = unnorm(a_c), unnorm(a_s), unnorm(a_m)
    def d(x, y):
        z = (x - y).abs()
        return {"xyz": float(z[..., :3].mean()), "rot": float(z[..., 3:6].mean()),
                "gripper": float(z[..., 6].mean()), "all": float(z.mean())}
    v_cs = (c - s).reshape(-1); v_ms = (m - s).reshape(-1); v_mc = (m - c).reshape(-1)
    denom = float(v_cs.norm() * v_ms.norm() + 1e-12)
    return {"shift": d(s, c), "repair": d(m, s), "to_clean": d(m, c),
            "align_cos": float(torch.dot(v_ms, v_cs) / denom),
            "align_proj": float(torch.dot(v_ms, v_cs) / (torch.dot(v_cs, v_cs) + 1e-12))}


def avg(records):
    out = {}
    for key, first in records[0].items():
        if isinstance(first, dict):
            out[key] = {sub: float(np.mean([r[key][sub] for r in records])) for sub in first}
        else:
            out[key] = float(np.mean([r[key] for r in records]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--clean-suite", default="libero_object")
    ap.add_argument("--clean-bddl", type=Path, required=True)
    ap.add_argument("--clean-init-file", type=Path, required=True)
    ap.add_argument("--clean-language", required=True)
    ap.add_argument("--target-view", default="agentview")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--z0-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    configure(args)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    manifest = json.loads((args.dir / "manifest.json").read_text())
    z = np.load(args.dir / "features.npz")
    valid = [r["index"] for r in manifest["records"] if r.get("compare_pass")]
    pos = {s: j for j, s in enumerate(valid)}
    dev = [i for i in manifest["dev_indices"] if i in pos]
    # E1 screening used the first 10 Dev states; keep them out of E2.
    remaining = dev[10:]
    fit_ids = remaining[: len(remaining) // 2]
    eval_ids = remaining[len(remaining) // 2:]
    views = manifest["policy_views"]
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
                if done: raise RuntimeError(f"settling done at {state}")
            states8[state] = state_from_observation(obs, policy.normalizer).astype(np.float32)
    finally:
        env.close()

    def patches(state, kind):
        j = pos[state]
        return {v: z[f"{kind}_{v}"][j] for v in views}

    def visual(state, kind):
        v, l, p = visual_tokens(policy, patches(state, kind), args.clean_language)
        return v, l, p

    # Fit visual residual subspace on fit states only.
    deltas = []
    for s in fit_ids:
        vc, _, _ = visual(s, "clean"); vs, _, _ = visual(s, "shift")
        deltas.append((vs - vc).reshape(-1).float().detach().cpu().numpy())
    D = np.stack(deltas, 0)
    delta_mean = D.mean(axis=0)
    _, sv, vt = np.linalg.svd(D.astype(np.float64), full_matrices=False)
    r = min(args.rank, vt.shape[0])
    U = vt[:r].T  # [D, r]
    rng = np.random.RandomState(args.seed)
    randoms = []
    for seed in range(3):
        Q, _ = np.linalg.qr(rng.normal(size=(D.shape[1], r)))
        randoms.append(Q)
    out = {"condition": str(args.dir), "fit_ids": fit_ids, "eval_ids": eval_ids,
           "rank": r, "singular_values": sv[:r].tolist(), "alphas": args.alphas,
           "per_state": [], "aggregates": {}}
    for state in eval_ids:
        vc, lang, pooled = visual(state, "clean")
        vs, _, _ = visual(state, "shift")
        state8 = states8[state]
        rep_c = repr_from_visual(policy, vc, lang, pooled, state8)
        rep_s = repr_from_visual(policy, vs, lang, pooled, state8)
        torch.manual_seed(args.seed + state)
        dev = next(policy.parameters()).device
        z0s = [torch.randn(1, policy.action_generator.action_horizon,
                           policy.action_generator.action_dim, device=dev)
               for _ in range(args.z0_count)]
        base = []
        for z0 in z0s:
            a_c = sample_with_z0(policy.action_generator, rep_c, z0)
            a_s = sample_with_z0(policy.action_generator, rep_s, z0)
            base.append((a_c, a_s))
        delta = (vs - vc).reshape(-1).detach().cpu().numpy()
        candidates = {
            "frozen": rep_s,
            "clean": rep_c,
            "mean": repr_from_visual(policy, vs - torch.as_tensor(delta_mean.reshape(vs.shape),
                                                                   device=vs.device, dtype=vs.dtype), lang, pooled, state8),
        }
        for alpha in args.alphas:
            proj = U @ (U.T @ delta)
            low = vs - alpha * torch.as_tensor(proj.reshape(vs.shape), device=vs.device, dtype=vs.dtype)
            rev = vs + alpha * torch.as_tensor(proj.reshape(vs.shape), device=vs.device, dtype=vs.dtype)
            candidates[f"lowrank_a{alpha}"] = repr_from_visual(policy, low, lang, pooled, state8)
            candidates[f"reverse_a{alpha}"] = repr_from_visual(policy, rev, lang, pooled, state8)
            for si, Q in enumerate(randoms):
                rp = Q @ (Q.T @ delta)
                vr = vs - alpha * torch.as_tensor(rp.reshape(vs.shape), device=vs.device, dtype=vs.dtype)
                candidates[f"random{si}_a{alpha}"] = repr_from_visual(policy, vr, lang, pooled, state8)
        rec = {"state": state, "split": "eval", "candidates": {}}
        for name, rep in candidates.items():
            vals = []
            for z0, (a_c, a_s) in zip(z0s, base):
                vals.append(action_metrics(a_c, a_s, sample_with_z0(policy.action_generator, rep, z0),
                                           policy.normalizer))
            rec["candidates"][name] = avg(vals)
        out["per_state"].append(rec)
    for name in out["per_state"][0]["candidates"]:
        out["aggregates"][name] = avg([r["candidates"][name] for r in out["per_state"]])
    out["note"] = "Oracle diagnostic: lowrank/reverse/random/mean use current Vc; not deployable."
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "e2_resampler_repair.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"fit": fit_ids, "eval": eval_ids, "rank": r}, indent=2))


if __name__ == "__main__":
    main()
