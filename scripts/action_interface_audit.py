"""Action-interface audit for paired style subspaces.

Fits a low-rank Shift-Clean subspace on development pairs, projects it out of
held-out Shift resampler features, and measures:
- Shift action discrepancy vs Clean action
- Clean action identity after applying the same projection

This is a privileged diagnostic; Clean pairs are not deployment inputs.
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
from target_only_action_validation import sample_z  # noqa: E402
from style_statistics_diagnostic import build_rep  # noqa: E402


def pca_basis(x, rank):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:min(rank, vt.shape[0])].T


def metrics(policy, pred, ref):
    pred = pred[:, :, :].float(); ref = ref.float()
    # reuse controller-space metric definitions
    p = policy.normalizer.unnormalize_action(pred).float().cpu().numpy()
    r = policy.normalizer.unnormalize_action(ref).float().cpu().numpy()
    pc = np.clip(p, -1, 1); rc = np.clip(r, -1, 1)
    from scipy.spatial.transform import Rotation
    pr = Rotation.from_rotvec(pc[..., 3:6].reshape(-1, 3))
    rr = Rotation.from_rotvec(rc[..., 3:6].reshape(-1, 3))
    return {"normalized_mae": float((pred - ref).abs().mean()),
            "xyz_mae_controller_units": float(np.abs(pc[..., :3] - rc[..., :3]).mean()),
            "rotation_geodesic_rad": float((pr.inv() * rr).magnitude().mean()),
            "gripper_mae": float(np.abs(pc[..., 6] - rc[..., 6]).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ranks", default="2,8")
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
    dev = [int(i) for i in manifest["dev_indices"] if int(i) in by_state]
    held = [int(i) for i in manifest["heldout_indices"] if int(i) in by_state]
    views = list(manifest["policy_views"])
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    _, pooled_lang = policy.representation_extractor.language_encoder([language])

    def get_visual(state, kind):
        patches = {v: torch.as_tensor(z[f"{kind}_{v}"][by_state[state]], device=device,
                                      dtype=torch.float32).unsqueeze(0) for v in views}
        with torch.no_grad():
            vis = policy.representation_extractor.cross_attention_resampler(
                patches, pooled_lang)[0]
        return vis.detach()

    clean_vis = {s: get_visual(s, "clean") for s in dev + held}
    shift_vis = {s: get_visual(s, "shift") for s in dev + held}
    K = clean_vis[dev[0]].shape[0] // len(views)
    records = {}
    for rank in [int(x) for x in args.ranks.split(",")]:
        basis = {}
        for vi, view in enumerate(views):
            d_dev = np.stack([
                (shift_vis[s][vi*K:(vi+1)*K].mean(0) -
                 clean_vis[s][vi*K:(vi+1)*K].mean(0)).cpu().numpy() for s in dev
            ])
            basis[view] = pca_basis(d_dev, rank)
        clean_mean = {
            view: torch.stack([clean_vis[s][vi*K:(vi+1)*K].mean(0) for s in dev]).mean(0)
            for vi, view in enumerate(views)
        }
        rows_out = []
        for state in held:
            clean = clean_vis[state]; shift = shift_vis[state]
            corr = torch.zeros_like(shift)
            for vi, view in enumerate(views):
                u = torch.as_tensor(basis[view], device=device, dtype=shift.dtype)
                d = shift[vi*K:(vi+1)*K].mean(0) - clean[vi*K:(vi+1)*K].mean(0)
                p = torch.matmul(torch.matmul(d, u), u.T)
                corr[vi*K:(vi+1)*K] = p.unsqueeze(0)
            state_t = np.asarray(z["state8"][by_state[state]], dtype=np.float32)
            clean_rep = build_rep(policy, clean.unsqueeze(0), language, state_t, device)
            shift_rep = build_rep(policy, shift.unsqueeze(0), language, state_t, device)
            corr_rep = build_rep(policy, (shift - corr).unsqueeze(0), language, state_t, device)
            gen = torch.Generator(device=device); gen.manual_seed(args.seed + state)
            z0 = torch.randn((1, policy.action_generator.action_horizon,
                              policy.action_generator.action_dim),
                             device=device, dtype=torch.float32, generator=gen)
            a_clean = sample_z(policy, clean_rep, z0)
            a_shift = sample_z(policy, shift_rep, z0)
            a_corr = sample_z(policy, corr_rep, z0)
            clean_corr = torch.zeros_like(clean)
            for vi, view in enumerate(views):
                u = torch.as_tensor(basis[view], device=device, dtype=clean.dtype)
                d_clean = clean[vi*K:(vi+1)*K].mean(0) - clean_mean[view]
                p_clean = torch.matmul(torch.matmul(d_clean, u), u.T)
                clean_corr[vi*K:(vi+1)*K] = p_clean.unsqueeze(0)
            clean_corr_rep = build_rep(policy, (clean - clean_corr).unsqueeze(0),
                                       language, state_t, device)
            a_clean_proj = sample_z(policy, clean_corr_rep, z0)
            rows_out.append({
                "index": state,
                "frozen": metrics(policy, a_shift, a_clean),
                "projected_shift": metrics(policy, a_corr, a_clean),
                "projection_norm": float(corr.norm()),
                "clean_identity": metrics(policy, a_clean_proj, a_clean),
            })
        records[str(rank)] = {
            "n_heldout": len(rows_out),
            "frozen_normalized_mae": float(np.mean([r["frozen"]["normalized_mae"] for r in rows_out])),
            "projected_normalized_mae": float(np.mean([r["projected_shift"]["normalized_mae"] for r in rows_out])),
            "frozen_xyz": float(np.mean([r["frozen"]["xyz_mae_controller_units"] for r in rows_out])),
            "projected_xyz": float(np.mean([r["projected_shift"]["xyz_mae_controller_units"] for r in rows_out])),
            "mean_projection_norm": float(np.mean([r["projection_norm"] for r in rows_out])),
            "clean_identity_normalized_mae": float(np.mean(
                [r["clean_identity"]["normalized_mae"] for r in rows_out])),
            "clean_identity_xyz": float(np.mean(
                [r["clean_identity"]["xyz_mae_controller_units"] for r in rows_out])),
            "records": rows_out,
        }
    payload = {"schema": "action_interface_audit_v1", "views": views,
               "records": records,
               "claim_scope": "privileged paired diagnostic; no closed-loop claim"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({r: {k: v for k, v in records[r].items() if k != "records"}
                      for r in records}, indent=2))


if __name__ == "__main__":
    main()
