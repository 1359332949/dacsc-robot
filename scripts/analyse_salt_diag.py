"""Aggregate Phase-1.2 SALT diagnostics (gate scores, beta saturation, loss curves)."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


def load(directory: str) -> list[dict]:
    return [json.load(open(f)) for f in sorted(glob.glob(f"{directory}/*.json"))]


def q(vals, p):
    return float(np.percentile(vals, p)) if vals else float("nan")


def main() -> None:
    groups = {
        "clean_always": load("/hy-tmp/validation/diag_clean_always"),
        "clean_feature": load("/hy-tmp/validation/diag_clean_feature"),
        "camera_feature": load("/hy-tmp/validation/diag_cam_feature"),
    }
    # ---- A. gate score distributions
    print("== A. gate scores (feature-gate runs)")
    for name in ("clean_feature", "camera_feature"):
        runs = groups[name]
        scores = [r["score"] for d in runs for r in d["replans"] if r["score"] is not None]
        thr = runs[0]["threshold"] if runs else None
        above = sum(1 for s in scores if thr is not None and s > thr)
        print(f"{name}: n_scores={len(scores)} threshold={thr} above={above} "
              f"min={q(scores,0):.3f} median={q(scores,50):.3f} p90={q(scores,90):.3f} "
              f"p95={q(scores,95):.3f} p99={q(scores,99):.3f} max={max(scores) if scores else float('nan'):.3f}")
    # first-replan vs later scores (camera has shift from episode start)
    for name in ("clean_feature", "camera_feature"):
        first = [d["replans"][0]["score"] for d in groups[name] if d["replans"][0]["score"] is not None]
        rest = [r["score"] for d in groups[name] for r in d["replans"][1:] if r["score"] is not None]
        print(f"{name}: first-replan median={q(first,50):.3f} rest median={q(rest,50):.3f}")

    # ---- B. beta saturation
    print("\n== B. beta saturation (always-adapt runs)")
    runs = groups["clean_always"]
    first_sat = []
    for d in runs:
        rep_betas = [r.get("beta") for r in d["replans"] if r.get("beta")]
        sat = next((i for i, b in enumerate(rep_betas) if b["frac_ge_099eta"] > 0.5), None)
        first_sat.append(sat)
        print(f"task={d['task_id']} seed={d['seed']}: saturation replan={sat} "
              f"beta_l2_final={rep_betas[-1]['beta_l2']:.3f} "
              f"frac099_final={rep_betas[-1]['frac_ge_099eta']:.3f}")
    # per-step update magnitude (first replan)
    print("\nfirst-replan update steps (delta_beta_l2, grad_l2, beta_l2, frac099):")
    for d in runs:
        first = [u for u in d["updates"] if u["replan"] == 0]
        for u in first:
            print(f"  task={d['task_id']} step={u['step']} dbeta={u['delta_beta_l2']:.3f} "
                  f"grad_l2={u['grad_l2']:.4f} beta_l2={u['beta_l2']:.3f} frac099={u['frac_ge_099eta']:.3f}")

    # ---- C. loss curves: training loss per step index, probe before/after per replan
    print("\n== C. loss")
    for d in runs:
        train = {}
        for u in d["updates"]:
            train.setdefault(u["step"], []).append(u["loss"])
        probe = [(r["probe_before"], r["probe_after"]) for r in d["replans"] if r.get("probe_before") is not None]
        pb = [p[0] for p in probe]; pa = [p[1] for p in probe]
        print(f"task={d['task_id']}: train_loss_per_step=" +
              ", ".join(f"{k}:{np.mean(v):.4f}" for k, v in sorted(train.items())))
        print(f"  probe_before first/last={pb[0]:.5f}/{pb[-1]:.5f}  probe_after first/last={pa[0]:.5f}/{pa[-1]:.5f}")


if __name__ == "__main__":
    main()
