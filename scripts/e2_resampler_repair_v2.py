"""E2 v2: corrected Resampler Oracle repair diagnostic.

Uses explicit feature_row/state8 mappings from v12 capture.  Reports paired-fit
mean, centered low-rank, mean+low-rank, norm-matched random, reverse controls.
All methods using eval Vc are Oracle; paired-fit mean uses fit Clean/Shift pairs.
"""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from corp import load_policy  # noqa: E402
from corp.representation.extractor import Representation  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@torch.no_grad()
def sample_with_z0(generator, representation, z0, steps=None):
    n = generator.num_sample_steps if steps is None else steps
    action = z0.to(device=representation.tokens.device, dtype=representation.action_dtype).clone()
    prefix = generator.prepare_prefix(representation)
    for step in range(n):
        tau = torch.full((action.shape[0],), step / float(n), device=action.device, dtype=action.dtype)
        action = action + generator(action, tau, prefix) * (1.0 / float(n))
    return action


def build_visual(policy, patches, instruction):
    ext = policy.representation_extractor
    device = next(policy.parameters()).device
    p = {v: torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
         for v, a in patches.items()}
    lang, pooled = ext.language_encoder(instruction)
    visual = ext.cross_attention_resampler(p, pooled)
    return visual, lang, pooled


def build_repr(policy, visual, lang, pooled, state8):
    device = next(policy.parameters()).device
    st = torch.as_tensor(state8, dtype=torch.float32, device=device).unsqueeze(0)
    st = policy.normalizer.normalize_state(st)
    state_token = policy.representation_extractor.state_encoder(st)
    tokens = torch.cat([visual, lang.to(visual.dtype), state_token], dim=1)
    return Representation(tokens=tokens, pooled_language=pooled, action_dtype=visual.dtype)


def dist(a, b, normalizer, prefix=6):
    def un(x): return normalizer.unnormalize_action(x)
    x, y = un(a), un(b)
    def one(m):
        z = (x[:, :m] - y[:, :m]).abs()
        return {"xyz": float(z[..., :3].mean()), "rot": float(z[..., 3:6].mean()),
                "gripper": float(z[..., 6].mean()), "all": float(z.mean())}
    return {"full": one(x.shape[1]), f"prefix{prefix}": one(min(prefix, x.shape[1]))}


def avg_metric(dicts):
    out = {}
    for key, val in dicts[0].items():
        if isinstance(val, dict):
            out[key] = {k: float(np.mean([d[key][k] for d in dicts])) for k in val}
        else:
            out[key] = float(np.mean([d[key] for d in dicts]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--clean-bddl", type=Path, required=True)
    ap.add_argument("--clean-init-file", type=Path, required=True)
    ap.add_argument("--clean-language", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--rank-grid", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--random-seeds", type=int, default=3)
    ap.add_argument("--z0-count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.dir / "manifest.json").read_text())
    z = np.load(args.dir / "features.npz")
    rows = [r for r in manifest["records"] if r.get("compare_pass") and r.get("feature_row") is not None]
    if not rows:
        raise SystemExit("no valid feature rows")
    # Hard validation of feature_row / state8 alignment.
    seen = set()
    for r in rows:
        row = int(r["feature_row"])
        if row in seen: raise SystemExit(f"duplicate feature_row {row}")
        seen.add(row)
        for v in manifest["policy_views"]:
            if row >= z[f"clean_{v}"].shape[0] or row >= z[f"shift_{v}"].shape[0]:
                raise SystemExit(f"feature_row {row} out of range")
    if "state8" not in z or z["state8"].shape[0] != z[f"clean_{manifest['policy_views'][0]}"].shape[0]:
        raise SystemExit("state8 missing or row count mismatch")
    by_state = {int(r["index"]): int(r["feature_row"]) for r in rows}
    dev = [i for i in manifest["dev_indices"] if i in by_state]
    remaining = dev[10:]
    rng = np.random.RandomState(args.split_seed)
    perm = rng.permutation(len(remaining))
    fit_ids = sorted(remaining[int(j)] for j in perm[:len(remaining)//2])
    eval_ids = sorted(remaining[int(j)] for j in perm[len(remaining)//2:])
    if not fit_ids or not eval_ids:
        raise SystemExit("empty fit/eval split")
    policy = load_policy(args.checkpoint, device=args.device); policy.eval()
    views = manifest["policy_views"]
    def feat(state, kind): return {v: z[f"{kind}_{v}"][by_state[state]] for v in views}
    def visual(state, kind):
        return build_visual(policy, feat(state, kind), args.clean_language)
    # Fit mean / centered PCA.
    deltas = []
    for s in fit_ids:
        vc, _, _ = visual(s, "clean"); vs, _, _ = visual(s, "shift")
        deltas.append((vs - vc).reshape(-1).detach().cpu().numpy().astype(np.float64))
    D = np.stack(deltas, 0); mean = D.mean(0); Dc = D - mean
    _, sv, vt = np.linalg.svd(Dc, full_matrices=False)
    ranks = [min(r, vt.shape[0]) for r in args.rank_grid]
    rng2 = np.random.RandomState(args.seed + 17)
    random_means = [rng2.normal(size=D.shape[1]) for _ in range(args.random_seeds)]
    U = {r: vt[:r].T for r in ranks}
    out = {"manifest_sha256": sha(args.dir / "manifest.json"),
           "features_sha256": sha(args.dir / "features.npz"),
           "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha(args.checkpoint),
           "script_sha256": sha(Path(__file__)), "manifest": manifest,
           "fit_ids": fit_ids, "eval_ids": eval_ids, "ranks": ranks,
           "singular_values": sv[:max(ranks)].tolist(),
           "alphas": args.alphas, "z0_count": args.z0_count,
           "information_budget": {
               "paired_fit_mean": "fit Clean/Shift pairs only; no eval Vc",
               "centered_lowrank": "fit U + eval Vc",
               "mean_lowrank": "fit mean/U + eval Vc",
               "random_mean": "fit mean norm only",
               "full_clean": "eval Vc"},
           "per_state": [], "aggregates": {}}
    for state in eval_ids:
        vc, lang, pooled = visual(state, "clean")
        vs, _, _ = visual(state, "shift")
        state8 = z["state8"][by_state[state]]
        rep_c = build_repr(policy, vc, lang, pooled, state8)
        rep_s = build_repr(policy, vs, lang, pooled, state8)
        dev = next(policy.parameters()).device
        torch.manual_seed(args.seed + state)
        z0s = [torch.randn(1, policy.action_generator.action_horizon,
                           policy.action_generator.action_dim, device=dev,
                           dtype=rep_c.action_dtype) for _ in range(args.z0_count)]
        base = []
        for z0 in z0s:
            ac = sample_with_z0(policy.action_generator, rep_c, z0)
            as_ = sample_with_z0(policy.action_generator, rep_s, z0)
            base.append((ac, as_))
        delta = (vs - vc).reshape(-1).detach().cpu().numpy().astype(np.float64)
        cands = {"frozen": rep_s, "full_clean": rep_c}
        for alpha in args.alphas:
            cands[f"paired_fit_mean_a{alpha}"] = build_repr(policy, vs - alpha * torch.as_tensor(
                mean.reshape(vs.shape), device=vs.device, dtype=vs.dtype), lang, pooled, state8)
            cands[f"reverse_mean_a{alpha}"] = build_repr(policy, vs + alpha * torch.as_tensor(
                mean.reshape(vs.shape), device=vs.device, dtype=vs.dtype), lang, pooled, state8)
            for si, rm in enumerate(random_means):
                rr = rm / (np.linalg.norm(rm) + 1e-12) * np.linalg.norm(mean)
                cands[f"random_mean{si}_a{alpha}"] = build_repr(policy, vs - alpha * torch.as_tensor(
                    rr.reshape(vs.shape), device=vs.device, dtype=vs.dtype), lang, pooled, state8)
            d = delta - mean
            for r in ranks:
                Ur = U[r]
                proj = Ur @ (Ur.T @ d)
                proj_t = torch.as_tensor(proj.reshape(vs.shape), device=vs.device, dtype=vs.dtype)
                mt = torch.as_tensor(mean.reshape(vs.shape), device=vs.device, dtype=vs.dtype)
                cands[f"centered_lowrank_r{r}_a{alpha}"] = build_repr(policy, vs - alpha * proj_t, lang, pooled, state8)
                cands[f"mean_lowrank_r{r}_a{alpha}"] = build_repr(policy, vs - alpha * mt - alpha * proj_t, lang, pooled, state8)
                cands[f"reverse_mean_lowrank_r{r}_a{alpha}"] = build_repr(policy, vs + alpha * mt + alpha * proj_t, lang, pooled, state8)
                ref = np.linalg.norm(proj)
                for si in range(args.random_seeds):
                    q = rng2.normal(size=d.shape[0])
                    rp = q / (np.linalg.norm(q) + 1e-12) * ref
                    cands[f"random_lowrank_r{r}_{si}_a{alpha}"] = build_repr(
                        policy, vs - alpha * torch.as_tensor(rp.reshape(vs.shape), device=vs.device, dtype=vs.dtype),
                        lang, pooled, state8)
        rec = {"state": state, "candidates": {}}
        for name, rep in cands.items():
            vals = []
            for z0, (ac, as_) in zip(z0s, base):
                am = sample_with_z0(policy.action_generator, rep, z0)
                vals.append(dist(am, ac, policy.normalizer))
            rec["candidates"][name] = avg_metric(vals)
        out["per_state"].append(rec)
    for name in out["per_state"][0]["candidates"]:
        out["aggregates"][name] = avg_metric([r["candidates"][name] for r in out["per_state"]])
    (args.out / "e2_resampler_repair_v2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"fit": fit_ids, "eval": eval_ids, "ranks": ranks}, indent=2))


if __name__ == "__main__":
    main()
