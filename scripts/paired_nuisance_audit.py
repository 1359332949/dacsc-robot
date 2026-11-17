"""Paired nuisance identifiability audit.

Uses privileged paired Clean/Shift captures to test whether a stable, low-rank
style subspace exists and whether normal Clean content leaks into it.  This is
an offline diagnostic; Clean pairs are never used by a deployment adapter.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos/CoRP"))
sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from corp import load_policy  # noqa: E402


def pca_basis(x: np.ndarray, rank: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("need at least two pairs")
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:min(rank, vt.shape[0])].T


def energy_fraction(x: np.ndarray, basis: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    denom = np.sum(x ** 2) + 1e-12
    return float(np.sum((x @ basis) ** 2) / denom)


def cosine_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1)
    keep = norms > 1e-12
    x = x[keep] / norms[keep, None]
    if len(x) < 2:
        return {"n": int(len(x)), "mean": None, "min": None}
    sims = x @ x.T
    iu = np.triu_indices(len(x), k=1)
    vals = sims[iu]
    return {"n": int(len(x)), "mean": float(vals.mean()), "min": float(vals.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ranks", default="2,4,8,16,32,64")
    ap.add_argument("--device", default="cuda:0")
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
    dev = [int(i) for i in manifest["dev_indices"] if int(i) in by_state]
    held = [int(i) for i in manifest["heldout_indices"] if int(i) in by_state]
    if not dev or not held:
        raise ValueError("need dev and heldout paired states")
    views = list(manifest["policy_views"])
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    _, pooled_lang = policy.representation_extractor.language_encoder([language])
    records = {"pair_integrity": {"n_dev": len(dev), "n_heldout": len(held),
                                 "compare_pass_only": True},
               "stages": {}, "views": {}}
    all_clean = {}
    all_shift = {}
    for state_id in dev + held:
        all_clean[state_id] = {
            v: torch.as_tensor(z[f"clean_{v}"][by_state[state_id]], device=device,
                               dtype=torch.float32).unsqueeze(0) for v in views}
        all_shift[state_id] = {
            v: torch.as_tensor(z[f"shift_{v}"][by_state[state_id]], device=device,
                               dtype=torch.float32).unsqueeze(0) for v in views}
    clean_vis = {s: policy.representation_extractor.cross_attention_resampler(
        all_clean[s], pooled_lang)[0].detach() for s in dev + held}
    shift_vis = {s: policy.representation_extractor.cross_attention_resampler(
        all_shift[s], pooled_lang)[0].detach() for s in dev + held}
    tokens_per_view = clean_vis[dev[0]].shape[0] // len(views)
    for view in views:
        view_idx = views.index(view)
        clean_p = {s: all_clean[s][view] for s in dev + held}
        shift_p = {s: all_shift[s][view] for s in dev + held}
        # DINO patch pooled and Resampler pooled are different stages.
        stages = {}
        clean_patch = {s: clean_p[s].mean(dim=1)[0].cpu().numpy() for s in dev + held}
        shift_patch = {s: shift_p[s].mean(dim=1)[0].cpu().numpy() for s in dev + held}
        clean_res = {s: clean_vis[s][view_idx * tokens_per_view:
                                      (view_idx + 1) * tokens_per_view].mean(dim=0).cpu().numpy()
                     for s in dev + held}
        shift_res = {s: shift_vis[s][view_idx * tokens_per_view:
                                      (view_idx + 1) * tokens_per_view].mean(dim=0).cpu().numpy()
                     for s in dev + held}
        for stage_name, c_map, s_map in (
            ("dino_patch_pooled", clean_patch, shift_patch),
            ("resampler_pooled", clean_res, shift_res),
        ):
            d_dev = np.stack([s_map[s] - c_map[s] for s in dev])
            d_held = np.stack([s_map[s] - c_map[s] for s in held])
            c_dev = np.stack([c_map[s] for s in dev])
            c_dev = c_dev - c_dev.mean(axis=0, keepdims=True)
            stage_metrics = {}
            for rank in [int(x) for x in args.ranks.split(",")]:
                basis = pca_basis(d_dev, rank)
                stage_metrics[str(rank)] = {
                    "style_energy_fraction_dev": energy_fraction(d_dev, basis),
                    "style_energy_fraction_heldout": energy_fraction(d_held, basis),
                    "content_leakage_clean_dev": energy_fraction(c_dev, basis),
                    "rank": int(basis.shape[1]),
                    "dim": int(d_dev.shape[1]),
                }
            stages[stage_name] = {
                "by_rank": stage_metrics,
                "style_cosine_dev": cosine_stats(d_dev),
                "style_cosine_heldout": cosine_stats(d_held),
            }
        records["views"][view] = stages
    records["interpretation"] = {
        "style_high_content_low": "possible separable nuisance subspace",
        "content_high": "style subspace overlaps normal task variation",
        "heldout_low": "subspace does not generalize",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
