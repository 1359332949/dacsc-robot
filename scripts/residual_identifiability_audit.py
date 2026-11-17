"""Offline audit: can source first-chunk residuals predict target residuals?

No closed-loop rollouts and no new adapter. All target Clean actions are
privileged diagnostic labels, never adapter inputs.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def unit(x):
    n = np.linalg.norm(x.ravel())
    return x.ravel() / (n + 1e-12)


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


def proj_metrics(pred, true):
    p = pred.ravel(); t = true.ravel()
    denom = float(np.dot(p, p)) + 1e-12
    lam = float(np.dot(p, t) / denom)
    orth = float(np.linalg.norm(t - lam * p) / (np.linalg.norm(t) + 1e-12))
    return lam, orth


def load_pairs(path):
    d = json.loads(Path(path).read_text())
    recs = d["records"]
    s8 = np.stack([np.asarray(r["domains"]["shift"]["state8"]) for r in recs])
    visual = np.stack([np.asarray(r["domains"]["shift"]["visual_mean"]) for r in recs])
    delta = np.stack([np.asarray(r["domains"]["clean"]["actions"]) - np.asarray(r["domains"]["shift"]["actions"])
                      for r in recs])
    clean_s8 = np.stack([np.asarray(r["domains"]["clean"]["state8"]) for r in recs])
    return d, recs, s8, clean_s8, visual, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-pairs", type=Path, required=True)
    ap.add_argument("--target-pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-results", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=2014)
    args = ap.parse_args()
    src_d, src_recs, src_s8, src_c8, src_vis, src_delta = load_pairs(args.source_pairs)
    tgt_d, tgt_recs, tgt_s8, tgt_c8, tgt_vis, tgt_delta = load_pairs(args.target_pairs)
    # Pairing sanity: only 8D proprio is available in this capture.
    pairing = {
        "source_state8_max_abs_diff": float(np.abs(src_s8 - src_c8).max()),
        "target_state8_max_abs_diff": float(np.abs(tgt_s8 - tgt_c8).max()),
        "source_clean_shift_action_shape_equal": bool(src_delta.shape[1:] == (12, 7)),
        "target_clean_shift_action_shape_equal": bool(tgt_delta.shape[1:] == (12, 7)),
        "note": "state8 equality is necessary, not sufficient; full MuJoCo/camera pairing not audited here",
    }
    # predictors
    global_pred = np.broadcast_to(src_delta.mean(0), tgt_delta.shape)
    # standardize each modality separately
    sm, ss = src_s8.mean(0), src_s8.std(0) + 1e-6
    vm, vs = src_vis.mean(0), src_vis.std(0) + 1e-6
    src_state_n = (src_s8 - sm) / ss
    tgt_state_n = (tgt_s8 - sm) / ss
    src_joint_n = np.concatenate([src_state_n, (src_vis - vm) / vs], axis=1)
    tgt_joint_n = np.concatenate([tgt_state_n, (tgt_vis - vm) / vs], axis=1)
    preds = {"zero": np.zeros_like(tgt_delta), "global_mean": global_pred}
    # [N_target, N_source]: each target retrieves its own source neighbours
    for metric, src_n, tgt_n in (
        ("state", src_state_n, tgt_state_n),
        ("joint", src_joint_n, tgt_joint_n),
    ):
        dist = np.linalg.norm(tgt_n[:, None, :] - src_n[None, :, :], axis=2)
        nn = np.argsort(dist, axis=1)
        for k in (1, 3, 5):
            preds[f"knn_{metric}_k{k}"] = np.stack([
                src_delta[nn[j, :k]].mean(0) for j in range(len(tgt_delta))
            ])
    # shuffled negative control: distribution over 5000 random source-residual pairings
    rng = np.random.default_rng(args.seed)
    shuffled_cosines = []
    for _ in range(5000):
        perm = rng.permutation(len(src_delta))[:len(tgt_delta)]
        shuffled_cosines.append(np.median([
            cos(src_delta[perm[j]], tgt_delta[j]) for j in range(len(tgt_delta))
        ]))
    shuffled_cosines = np.asarray(shuffled_cosines)
    preds["shuffled"] = src_delta[rng.permutation(len(src_delta))[:len(tgt_delta)]]
    # success labels optional
    success = {}
    if args.target_results and Path(args.target_results).exists():
        rd = json.loads(Path(args.target_results).read_text())
        for r in rd.get("records", []):
            if r.get("domain") == "shift":
                success[int(r["state_id"])] = bool(r["success"])
    report = {"schema": "residual_identifiability_audit_v2", "pairing": pairing,
              "source_n": len(src_delta), "target_n": len(tgt_delta),
              "shuffled_cosine_median_distribution": {
                  "n": int(len(shuffled_cosines)),
                  "median": float(np.median(shuffled_cosines)),
                  "p05": float(np.percentile(shuffled_cosines, 5)),
                  "p95": float(np.percentile(shuffled_cosines, 95)),
              },
              "predictors": {}}
    for name, pred in preds.items():
        per = []
        for j, tid in enumerate([int(r["state_id"]) for r in tgt_recs]):
            c = cos(pred[j], tgt_delta[j]); lam, orth = proj_metrics(pred[j], tgt_delta[j])
            per.append({"state_id": tid, "cosine": c, "scale": lam, "orthogonal_error": orth,
                        "success": success.get(tid)})
        cs = np.array([x["cosine"] for x in per]); os_ = np.array([x["orthogonal_error"] for x in per])
        dim_cos = [cos(pred[:, :, d], tgt_delta[:, :, d]) for d in range(7)]
        step_cos = [cos(pred[:, t, :], tgt_delta[:, t, :]) for t in range(12)]
        report["predictors"][name] = {
            "cosine_median": float(np.median(cs)), "cosine_mean": float(cs.mean()),
            "cosine_min": float(cs.min()), "cosine_max": float(cs.max()),
            "orthogonal_error_median": float(np.median(os_)),
            "per_state": per, "per_dim_cosine": dim_cos, "per_step_cosine": step_cos,
        }
    # privileged reference: target residual mean direction vs source mean direction
    report["source_mean_norm"] = float(np.linalg.norm(src_delta.mean(0)))
    report["target_mean_norm"] = float(np.linalg.norm(tgt_delta.mean(0)))
    report["source_vs_target_mean_cosine"] = cos(src_delta.mean(0), tgt_delta.mean(0))
    report["target_residual_cos_to_source_global_mean"] = [
        cos(src_delta.mean(0), tgt_delta[j]) for j in range(len(tgt_delta))
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
