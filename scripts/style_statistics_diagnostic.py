"""Privileged style-statistics diagnostic for CoRP features.

Tests the hypothesis that visual style is captured by per-view channel mean/std
of DINO patch tokens, while content is the normalized spatial pattern.  Clean
and Shift are paired offline; this is a diagnostic, not a deployment method.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos/CoRP"))
sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))

from corp import load_policy  # noqa: E402
from target_only_action_validation import sample_z, _one_action_metrics  # noqa: E402


def prepare_patches(z, view, kind, row, device):
    return torch.as_tensor(z[f"{kind}_{view}"][row], device=device,
                          dtype=torch.float32).unsqueeze(0)


def adain(src, ref):
    """Transfer per-channel statistics of ref onto src."""
    mean_s = src.mean(dim=1, keepdim=True)
    std_s = src.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    mean_r = ref.mean(dim=1, keepdim=True)
    std_r = ref.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (src - mean_s) / std_s * std_r + mean_r


def adain_stats(src, mean_r, std_r):
    mean_s = src.mean(dim=1, keepdim=True)
    std_s = src.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (src - mean_s) / std_s * std_r + mean_r


def mean_only(src, mean_r):
    return src - src.mean(dim=1, keepdim=True) + mean_r


def std_only(src, std_r):
    mean_s = src.mean(dim=1, keepdim=True)
    std_s = src.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (src - mean_s) / std_s * std_r + mean_s


def build_rep(policy, visual, language, state, device):
    lang_tokens, pooled = policy.representation_extractor.language_encoder([language])
    state_t = torch.as_tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    state_token = policy.representation_extractor.state_encoder(state_t)
    return SimpleNamespace(tokens=torch.cat([visual, lang_tokens, state_token], dim=1),
                           pooled_language=pooled, action_dtype=visual.dtype)


def act(policy, visual, rep, z0):
    return sample_z(policy, rep, z0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == "libero_object")
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == 2222)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    manifest = json.loads((args.capture / "manifest.json").read_text())
    z = np.load(args.capture / "features.npz")
    rows = [r for r in manifest["records"] if r.get("compare_pass")]
    by_state = {int(r["index"]): int(r["feature_row"]) for r in rows}
    views = list(manifest["policy_views"])
    dev_ids = [int(i) for i in manifest["dev_indices"] if int(i) in
               {int(r["index"]): int(r["feature_row"]) for r in rows}]
    source_fit = dev_ids[10:20]
    source_mean = {}
    source_std = {}
    for v in views:
        vals = torch.stack([
            torch.as_tensor(z[f"clean_{v}"][int(next(r["feature_row"] for r in rows
                                                     if int(r["index"]) == s))],
                            dtype=torch.float32)
            for s in source_fit
        ])
        source_mean[v] = vals.mean(dim=(0, 1), keepdim=True)
        source_std[v] = vals.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(1e-6)
    device = torch.device(args.device)
    source_mean = {k: v.to(device) for k, v in source_mean.items()}
    source_std = {k: v.to(device) for k, v in source_std.items()}
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    source_res_mean = {}
    source_res_std = {}
    for view in views:
        vals = []
        for s in source_fit:
            patches = {v: torch.as_tensor(z[f"clean_{v}"][by_state[s]], device=args.device,
                                         dtype=torch.float32).unsqueeze(0) for v in views}
            _, pooled = policy.representation_extractor.language_encoder([language])
            vis = policy.representation_extractor.cross_attention_resampler(patches, pooled)
            vals.append(vis[0].detach().cpu())
        vals = torch.stack(vals)
        source_res_mean[view] = vals.mean(dim=(0, 1), keepdim=True)
        source_res_std[view] = vals.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(1e-6)
    source_res_mean = {k: v.to(device) for k, v in source_res_mean.items()}
    source_res_std = {k: v.to(device) for k, v in source_res_std.items()}
    records = []
    for r in rows:
        row = int(r["feature_row"]); state = np.asarray(z["state8"][row], dtype=np.float32)
        patches_clean = {v: prepare_patches(z, v, "clean", row, device) for v in views}
        patches_shift = {v: prepare_patches(z, v, "shift", row, device) for v in views}
        _, pooled_lang = policy.representation_extractor.language_encoder([language])
        visual_clean = policy.representation_extractor.cross_attention_resampler(patches_clean, pooled_lang)
        visual_shift = policy.representation_extractor.cross_attention_resampler(patches_shift, pooled_lang)
        adain_patches = {v: adain(patches_shift[v], patches_clean[v]) for v in views}
        visual_adain = policy.representation_extractor.cross_attention_resampler(adain_patches, pooled_lang)
        source_patches = {
            v: adain_stats(patches_shift[v], source_mean[v], source_std[v])
            for v in views
        }
        visual_adain_source = policy.representation_extractor.cross_attention_resampler(
            source_patches, pooled_lang)
        # Post-Resampler intervention: statistics of the 96 output tokens.
        tok_mean = visual_shift.mean(dim=1, keepdim=True)
        tok_std = visual_shift.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        tokens_per_view = visual_shift.shape[1] // len(views)
        ref_mean = torch.cat([
            source_res_mean[v].expand(-1, tokens_per_view, -1) for v in views], dim=1)
        ref_std = torch.cat([
            source_res_std[v].expand(-1, tokens_per_view, -1) for v in views], dim=1)
        visual_res_mean = visual_shift - tok_mean + ref_mean
        visual_res_adain = (visual_shift - tok_mean) / tok_std * ref_std + ref_mean
        mean_patches = {v: mean_only(patches_shift[v], source_mean[v]) for v in views}
        visual_mean_only = policy.representation_extractor.cross_attention_resampler(
            mean_patches, pooled_lang)
        std_patches = {v: std_only(patches_shift[v], source_std[v]) for v in views}
        visual_std_only = policy.representation_extractor.cross_attention_resampler(
            std_patches, pooled_lang)
        clean_mean_patches = {v: mean_only(patches_clean[v], source_mean[v]) for v in views}
        visual_clean_mean = policy.representation_extractor.cross_attention_resampler(
            clean_mean_patches, pooled_lang)
        clean_adain_patches = {
            v: adain_stats(patches_clean[v], source_mean[v], source_std[v]) for v in views}
        visual_clean_adain = policy.representation_extractor.cross_attention_resampler(
            clean_adain_patches, pooled_lang)
        # style statistics per view (channel mean/std over patch axis)
        stats = {}
        for v in views:
            c = patches_clean[v][0].float(); s = patches_shift[v][0].float()
            stats[v] = {
                "clean_mean_norm": float(c.mean(0).norm()),
                "shift_mean_norm": float(s.mean(0).norm()),
                "mean_diff_norm": float((s.mean(0) - c.mean(0)).norm()),
                "clean_std_norm": float(c.std(0, unbiased=False).norm()),
                "shift_std_norm": float(s.std(0, unbiased=False).norm()),
                "std_diff_norm": float((s.std(0, unbiased=False) - c.std(0, unbiased=False)).norm()),
            }
        rep_clean = build_rep(policy, visual_clean, language, state, device)
        rep_shift = build_rep(policy, visual_shift, language, state, device)
        rep_adain = build_rep(policy, visual_adain, language, state, device)
        rep_adain_source = build_rep(policy, visual_adain_source, language, state, device)
        rep_mean_only = build_rep(policy, visual_mean_only, language, state, device)
        rep_res_mean = build_rep(policy, visual_res_mean, language, state, device)
        rep_res_adain = build_rep(policy, visual_res_adain, language, state, device)
        rep_std_only = build_rep(policy, visual_std_only, language, state, device)
        rep_clean_mean = build_rep(policy, visual_clean_mean, language, state, device)
        rep_clean_adain = build_rep(policy, visual_clean_adain, language, state, device)
        gen = torch.Generator(device=device); gen.manual_seed(args.seed + int(r["index"]))
        z0 = torch.randn((1, policy.action_generator.action_horizon,
                          policy.action_generator.action_dim),
                         device=device, dtype=torch.float32, generator=gen)
        a_clean = act(policy, visual_clean, rep_clean, z0)
        a_shift = act(policy, visual_shift, rep_shift, z0)
        a_adain = act(policy, visual_adain, rep_adain, z0)
        a_adain_source = act(policy, visual_adain_source, rep_adain_source, z0)
        a_mean_only = act(policy, visual_mean_only, rep_mean_only, z0)
        a_res_mean = act(policy, visual_res_mean, rep_res_mean, z0)
        a_res_adain = act(policy, visual_res_adain, rep_res_adain, z0)
        a_std_only = act(policy, visual_std_only, rep_std_only, z0)
        a_clean_mean = act(policy, visual_clean_mean, rep_clean_mean, z0)
        a_clean_adain = act(policy, visual_clean_adain, rep_clean_adain, z0)
        records.append({
            "index": int(r["index"]),
            "style_stats": stats,
            "frozen": _one_action_metrics(policy, a_shift, a_clean, a_clean.shape[1]),
            "adain": _one_action_metrics(policy, a_adain, a_clean, a_clean.shape[1]),
            "adain_source_stats": _one_action_metrics(
                policy, a_adain_source, a_clean, a_clean.shape[1]),
            "mean_only": _one_action_metrics(policy, a_mean_only, a_clean, a_clean.shape[1]),
            "resampler_mean_only": _one_action_metrics(
                policy, a_res_mean, a_clean, a_clean.shape[1]),
            "resampler_adain": _one_action_metrics(
                policy, a_res_adain, a_clean, a_clean.shape[1]),
            "std_only": _one_action_metrics(policy, a_std_only, a_clean, a_clean.shape[1]),
            "oracle_clean": _one_action_metrics(policy, a_clean, a_clean, a_clean.shape[1]),
            "clean_identity_mean_only": _one_action_metrics(
                policy, a_clean_mean, a_clean, a_clean.shape[1]),
            "clean_identity_adain_source": _one_action_metrics(
                policy, a_clean_adain, a_clean, a_clean.shape[1]),
        })
    methods = ("frozen", "adain", "adain_source_stats", "mean_only", "std_only",
               "resampler_mean_only", "resampler_adain", "oracle_clean")
    keys = ("normalized_mae", "xyz_mae_controller_units", "rotation_geodesic_rad",
            "gripper_mae", "action_saturation_fraction")
    summary = {m: {k: float(np.mean([r[m][k] for r in records])) for k in keys} for m in methods}
    clean_identity = {m: {k: float(np.mean([r[m][k] for r in records])) for k in keys}
                      for m in ("clean_identity_mean_only", "clean_identity_adain_source")}
    payload = {"schema": "style_statistics_diagnostic_v1", "language": language,
               "clean_identity": clean_identity,
               "views": views, "records": records, "summary": summary,
               "claim_scope": "privileged paired diagnostic; AdaIN uses Clean statistics"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
