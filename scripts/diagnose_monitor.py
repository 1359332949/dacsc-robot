"""Diagnose monitor false alarms: source-fixed threshold, domain confound."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from dacsc.failure_monitor import CausalFailureMonitor


def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg): return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def prep(d, ck):
    mu = {k: np.asarray(v[0]) for k, v in ck["normalization"].items()}
    sd = {k: np.asarray(v[1]) for k, v in ck["normalization"].items()}
    return {k: ((d[k] - mu[k]) / sd[k]).astype(np.float32) for k in ("state", "action", "visual_delta")}


def predict(model, x):
    with torch.no_grad():
        return model(torch.from_numpy(x["state"]), torch.from_numpy(x["action"]),
                     torch.from_numpy(x["visual_delta"])).numpy()


def ridge_domain(x, y, lam=1.0):
    # x: [N, D], y: {0,1}; closed-form ridge classification score.
    X = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1)
    yy = (2 * y - 1).astype(np.float32)
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ yy)
    return X @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    s = np.load(args.source, allow_pickle=True); t = np.load(args.target, allow_pickle=True)
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = CausalFailureMonitor(window_size=ck["window"], hidden_dim=ck["hidden"],
                                 use_visual=ck.get("use_visual", True))
    model.load_state_dict(ck["model"]); model.eval()
    val_groups = set(ck.get("val_groups", []))
    # source validation threshold
    s_val = np.array([g in val_groups for g in s["group"]])
    ps_val = predict(model, prep(s, ck))
    src_neg = np.sort(ps_val[s_val][s["y"][s_val] == 0])
    if len(src_neg) == 0: raise ValueError("no source validation negatives")
    thr = float(np.quantile(src_neg, 0.95))
    pt = predict(model, prep(t, ck))
    # domain classifier using flattened mean window features
    keys = ("state", "action") + (("visual_delta",) if ck.get("use_visual", True) else ())
    def flat(d):
        return np.concatenate([d[k].mean(axis=1) for k in keys], axis=1)
    src_x, tgt_x = flat(prep(s, ck)), flat(prep(t, ck))
    src_dom = (s["domain"] == "shift").astype(np.int64)
    # source-only domain classifier; standardize on source
    mu, sd = src_x.mean(0), src_x.std(0) + 1e-6
    src_dom_score = ridge_domain((src_x - mu) / sd, src_dom)
    tgt_dom_score = ridge_domain((tgt_x - mu) / sd, (t["domain"] == "shift").astype(np.int64))
    # target domain AUC is evaluated with target labels, diagnostic only.
    report = {
        "schema": "monitor_false_alarm_diagnosis_v1",
        "source_val_threshold": thr,
        "source_val_auc": auc(s["y"][s_val], ps_val[s_val]),
        "target_auc": auc(t["y"], pt),
        "target_at_source_threshold": {
            "recall": float(np.mean(pt[t["y"] == 1] >= thr)) if np.any(t["y"] == 1) else None,
            "false_alarm_episode_fraction": None,
        },
        "target_by_domain": {},
        "domain_classifier": {"source_auc": auc(src_dom, src_dom_score),
                              "target_auc": auc((t["domain"] == "shift"), tgt_dom_score)},
    }
    false_eps = []
    for dom in ("clean", "shift"):
        m = t["domain"] == dom
        if m.any():
            report["target_by_domain"][dom] = {
                "n_windows": int(m.sum()), "auc": auc(t["y"][m], pt[m]),
                "recall_at_source_thr": float(np.mean(pt[m & (t["y"] == 1)] >= thr)) if np.any(m & (t["y"] == 1)) else None,
            }
    for e in sorted(set(t["episode"].tolist())):
        idx = np.where(t["episode"] == e)[0]
        if t["y"][idx].max() < 0.5 and np.any(pt[idx] >= thr):
            false_eps.append({"episode": e, "domain": str(t["domain"][idx[0]]),
                              "state": str(t["group"][idx[0]]),
                              "first_alarm_step": int(t["time"][idx][pt[idx] >= thr][0]),
                              "max_score": float(pt[idx].max())})
    report["false_alarm_episodes"] = false_eps
    report["target_at_source_threshold"]["false_alarm_episode_fraction"] = (
        len(false_eps) / max(1, len(set(t["episode"].tolist())) - len(set(t["episode"][t["y"] > .5].tolist())))
    )
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
