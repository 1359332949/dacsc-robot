"""Offline pre-screen of three counterfactual targets after RNG fixes (no rollouts).

Targets: medoid incl identity, medoid excl identity, anchored mean alpha=0.1.
For each: medoid-identity frequency, target-identity distance, initial distill loss,
gradient norm, projected update norm (retain), and action change after one projected
step -- all with per-replan action/view seeds.
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
from dacsc.leftover_tta import sample_with_grad  # noqa: E402
from dacsc.sg_cpa import SGCpa, action_distance  # noqa: E402
from sg_cpa_phase4_final import CKPT, A_FILE, collect  # noqa: E402

OUT = Path("/hy-tmp/validation/sg_cpa_prescreen")
DEV = "cuda:0"


def sample_identity(policy, prepared, beta, seed):
    with torch.no_grad():
        policy.action_generator.blocks[0].attention.qkv  # noqa: B018
        torch.manual_seed(seed)
        rep = policy.representation_extractor(prepared["images"], prepared["state"],
                                              prepared["instruction"])
        return policy.action_generator.sample(rep).detach()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
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
    modes = [("medoid_incl", dict(aggregation="medoid", medoid_include_identity=True)),
             ("medoid_excl", dict(aggregation="medoid", medoid_include_identity=False)),
             ("anchored_mean", dict(aggregation="mean_anchored", alpha=0.1))]
    results = defaultdict(list)
    t0 = time.time()
    for mode, kw in modes:
        obj = SGCpa(policy, a_basis, seed=1234, **kw)
        try:
            for bi, b in enumerate(batches):
                prep = b["prepared"]
                for r in range(args.seeds):
                    action_seed = 5000 + 7919 * r + bi
                    view_seed = action_seed + 1
                    info = obj.build_target(prep, seed=action_seed, view_seed=view_seed)
                    ident = info["acts"][0]
                    target_dist = action_distance(info["target"], ident)
                    # real update path (projection + beta bound + A guard + rollback)
                    obj.adapter.beta.detach().zero_()
                    a_old = sample_identity(policy, prep, obj.adapter.beta, action_seed)
                    rec = obj.step(prep, seed=action_seed, view_seed=view_seed,
                                   target_info=info)
                    a_new = sample_identity(policy, prep, obj.adapter.beta, action_seed)
                    loss = torch.tensor(rec["loss"])
                    g_norm = rec["g_norm"]
                    projected_norm = g_norm * rec["retained_fraction"]
                    da_norm = float((a_new - a_old).abs().mean())
                    a0u = policy.normalizer.unnormalize_action(a_old)
                    a1u = policy.normalizer.unnormalize_action(a_new)
                    da_ctrl = float((a1u - a0u).abs().mean())
                    with torch.no_grad():
                        obj.adapter.beta.zero_()
                    rec = {"mode": mode, "domain": b["domain"], "batch": bi, "replan": r,
                           "medoid_index": info["medoid_index"],
                           "medoid_identity": float(info["medoid_index"] == 0
                                                    if mode.startswith("medoid") else 0.0),
                           "target_distance": target_dist,
                           "loss": float(loss),
                           "grad_norm": float(g_norm),
                           "projected_norm": float(projected_norm),
                           "retain": float(rec["retained_fraction"]),
                           "action_change": da_norm,
                           "action_change_ctrl": da_ctrl,
                           "accepted": bool(rec["accepted"]),
                           "backtracks": int(rec["backtracks"]),
                           "beta_norm": float(rec["d_beta_norm"])}
                    results[mode].append(rec)
        finally:
            obj.close()
        print(mode, "done", len(results[mode]), flush=True)
    agg = {}
    for mode, rs in results.items():
        agg[mode] = {}
        for dom in sorted({r["domain"] for r in rs}):
            sel = [r for r in rs if r["domain"] == dom]
            agg[mode][dom] = {k: {"mean": float(np.mean([r[k] for r in sel])),
                                  "std": float(np.std([r[k] for r in sel]))}
                              for k in ("medoid_identity", "target_distance", "loss",
                                        "grad_norm", "projected_norm", "retain",
                                        "action_change", "beta_norm")}
    (args.out / "prescreen.json").write_text(json.dumps({"aggregate": agg,
                                                         "raw_count": len(results[next(iter(results))])}))
    for mode in agg:
        print("==", mode)
        for dom, v in agg[mode].items():
            print(f"  {dom:11s} ident={v['medoid_identity']['mean']:.3f} "
                  f"target_d={v['target_distance']['mean']:.5f} loss={v['loss']['mean']:.6f} "
                  f"grad={v['grad_norm']['mean']:.4f} proj={v['projected_norm']['mean']:.5f} "
                  f"retain={v['retain']['mean']:.3f} da={v['action_change']['mean']:.6f} "
                  f"beta={v['beta_norm']['mean']:.5f}")
    print("seconds", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
