"""Offline eta calibration for SG-CPA (real SGCpa.step path, no robot rollouts).

For each fixed observation and eta: reset beta=0, run 10 projected SGD steps on a
FIXED teacher target/views/latent, record loss trajectory, raw/accepted dbeta,
beta saturation, action changes (normalized + controller units), D_A, guard stats.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import configure_libero  # noqa: E402
from corp import load_policy  # noqa: E402
from dacsc.sg_cpa import SGCpa  # noqa: E402
from sg_cpa_phase4_final import CKPT, A_FILE, collect  # noqa: E402

OUT = Path("/hy-tmp/validation/sg_cpa_eta_cal")
DEV = "cuda:0"
ETAS = (0.05, 0.5, 5.0, 20.0, 50.0, 100.0)
CHECKPOINTS = (1, 3, 5, 10)


def fork():
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    return torch.random.fork_rng(devices=devices)


def sample_view(policy, images, state, instruction, seed):
    with fork():
        torch.manual_seed(int(seed))
        with torch.no_grad():
            rep = policy.representation_extractor(images, state, instruction)
            return policy.action_generator.sample(rep).detach()


def action_change(policy, prepared, images, seed):
    a = sample_view(policy, images, prepared["state"], prepared["instruction"], seed)
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--etas", default="0.05,0.5,5,20,50,100")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    class C: pass
    c = C(); c.libero_src = Path("/hy-tmp/libero_plus")
    c.libero_config_path = Path("/hy-tmp/libero_plus_config")
    c.render_gpu = 0; c.device = DEV
    configure_libero(c)
    policy = load_policy(CKPT, device=DEV)
    a_basis = torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float64)
    batches = collect(policy, n_tasks=args.tasks, n_chunks=args.chunks)
    print(f"batches={len(batches)}", flush=True)
    etas = [float(x) for x in args.etas.split(",")]
    # fixed teacher targets per observation (built once, reused for every eta)
    targets = []
    for bi, b in enumerate(batches):
        obj = SGCpa(policy, a_basis, seed=1234)
        try:
            info = obj.build_target(b["prepared"], seed=90000 + bi, view_seed=90000 + bi + 1)
        finally:
            obj.close()
        targets.append(info)
    rows = []
    t0 = time.time()
    for eta in etas:
        for bi, b in enumerate(batches):
            prep = b["prepared"]
            seed = 90000 + bi
            vseed = seed + 1
            info = targets[bi]
            a0 = sample_view(policy, prep["images"], prep["state"], prep["instruction"], seed)
            views0 = [sample_view(policy, im, prep["state"], prep["instruction"], seed)
                      for im in info["views"]]
            obj = SGCpa(policy, a_basis, eta=eta, seed=1234)
            try:
                obj.adapter.beta.detach().zero_()
                last_loss = float(obj.loss(prep, info, seed=seed).detach())
                for step in range(1, args.steps + 1):
                    rec = obj.step(prep, seed=seed, view_seed=vseed, target_info=info)
                    row = {"eta": eta, "batch": bi, "step": step,
                           "loss": rec["loss"], "loss_prev": last_loss,
                           "loss_drop": last_loss - rec["loss"],
                           "raw_dbeta": rec["raw_d_beta_norm"],
                           "accepted_dbeta": rec["d_beta_norm"],
                           "accepted": int(rec["accepted"]), "backtracks": rec["backtracks"],
                           "beta_norm": rec["beta_norm"], "beta_abs_max": rec["beta_abs_max"],
                           "beta_sat_frac": float((obj.adapter.beta.detach().abs()
                                                   >= 0.95 * obj.adapter.eta).float().mean()),
                           "retain": rec["retained_fraction"], "g_norm": rec["g_norm"],
                           "d_a_local": rec["local_a_change"],
                           "d_a_total": rec["cumulative_a_change"]}
                    last_loss = rec["loss"]
                    if step in CHECKPOINTS:
                        a1 = sample_view(policy, prep["images"], prep["state"],
                                         prep["instruction"], seed)
                        da_norm = float((a1 - a0).abs().mean())
                        a0u = policy.normalizer.unnormalize_action(a0)
                        a1u = policy.normalizer.unnormalize_action(a1)
                        da_ctrl = float((a1u - a0u).abs().mean())
                        view_deltas = []
                        for im, av0 in zip(info["views"], views0):
                            av1 = sample_view(policy, im, prep["state"],
                                              prep["instruction"], seed)
                            view_deltas.append(float((av1 - av0).abs().mean()))
                        row.update({"da_identity_norm": da_norm, "da_identity_ctrl": da_ctrl,
                                    "da_views_norm": float(np.mean(view_deltas))})
                    rows.append(row)
            finally:
                obj.close()
        print("eta", eta, "done", flush=True)
    (args.out / "raw_rows.json").write_text(json.dumps(rows))
    agg = {}
    for eta in etas:
        agg[str(eta)] = {}
        for ck in CHECKPOINTS:
            sel = [r for r in rows if r["eta"] == eta and r["step"] == ck]
            agg[str(eta)][str(ck)] = {k: float(np.mean([r[k] for r in sel]))
                                      for k in ("loss", "loss_drop", "raw_dbeta",
                                                "accepted_dbeta", "accepted", "backtracks",
                                                "beta_norm", "beta_abs_max", "beta_sat_frac",
                                                "retain", "g_norm", "d_a_local", "d_a_total",
                                                "da_identity_norm", "da_identity_ctrl",
                                                "da_views_norm")}
    (args.out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps({eta: {ck: {k: round(v, 6) for k, v in agg[str(eta)][str(ck)].items()
                                  if k in ("loss_drop", "accepted_dbeta", "beta_abs_max",
                                           "da_identity_norm", "da_identity_ctrl",
                                           "da_views_norm", "accepted")}
                               for ck in ("1", "10")}
                      for eta in etas}, indent=1))
    print("seconds", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
