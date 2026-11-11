"""Final 4.0 offline validation: 4 domains, medoid/mean, K sweep, no-op, flow, Jacobian."""

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
sys.path.insert(0, str(ROOT / "repos" / "CoRP" / "scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.leftover_tta import sample_with_grad  # noqa: E402
from dacsc.salt_tta import flow_matching_leftover_loss  # noqa: E402
from dacsc.sg_cpa import (  # noqa: E402
    SGCpa, _noise_fn, action_distance, build_views, group_distances, medoid_index,
)

CKPT = ROOT / "repos" / "CoRP" / "checkpoints" / "corp_libero.pt"
A_FILE = Path("/hy-tmp/validation/interface_stats_spatial_k32_big2.json")
OUT = Path("/hy-tmp/validation/sg_cpa_4_0_final")
DEV = "cuda:0"
CELLS = json.loads((ROOT / "configs/phase3_final_tasks.json").read_text())["cells"]


def collect(policy, n_tasks=2, n_chunks=4, seed=7):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in
              ("libero_spatial", "libero_object")}
    batches = []
    for cell, info in CELLS.items():
        domain, suite = info["domain"], info["suite"]
        if domain not in ("Camera", "Background", "Noise", "Light"):
            continue
        suite_obj = suites[suite]
        for tid in info["tasks"][:n_tasks]:
            task = suite_obj.get_task(tid)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=0)
            np.random.seed(seed); torch.manual_seed(seed); env.seed(seed); env.reset()
            obs = env.set_init_state(states[0])
            for _ in range(SETTLING_STEPS):
                obs, _, done, _ = env.step(DUMMY_ACTION)
                if done: break
            for k in range(n_chunks):
                raw = policy_observation(obs, task.language, policy)
                prepared = prepare_observation(raw, config=policy.config, normalizer=policy.normalizer,
                                               device=torch.device(DEV))
                batches.append({"domain": domain, "suite": suite, "task_id": tid, "chunk": k,
                                "prepared": prepared})
                with torch.no_grad():
                    rep = policy.representation_extractor(**prepared)
                    chunk = policy.action_generator.sample(rep)
                actions = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
                for a in actions[:6]:
                    obs, _, done, _ = env.step(np.clip(a, -1.0, 1.0))
                    if done: break
                if done: break
            env.close()
    return batches


def metrics_for_target(info):
    acts = info["acts"]
    dview = float(np.mean([action_distance(acts[i], acts[j])
                           for i in range(len(acts)) for j in range(i + 1, len(acts))]))
    med_anchor = action_distance(info["target"], acts[0])
    mean_target = acts[0] + 0.1 * (torch.stack(acts[1:]).mean(dim=0) - acts[0])
    return {"D_view": dview, "medoid_index": info["medoid_index"],
            "D_anchor_medoid": med_anchor,
            "D_anchor_mean_anchored_0.1": action_distance(mean_target, acts[0]),
            "groups": info["groups"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    class C: pass
    c = C(); c.libero_src = Path("/hy-tmp/libero_plus")
    c.libero_config_path = Path("/hy-tmp/libero_plus_config"); c.render_gpu = 0; c.device = DEV
    configure_libero(c)
    policy = load_policy(CKPT, device=DEV)
    a_basis = torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float64)

    t0 = time.time()
    batches = collect(policy)
    print(f"collected {len(batches)} batches in {time.time()-t0:.1f}s", flush=True)

    obj = SGCpa(policy, a_basis, seed=1234)
    per_batch, k_sweep = [], defaultdict(list)
    try:
        for bi, b in enumerate(batches):
            prep = b["prepared"]
            t = time.time()
            info = obj.build_target(prep, seed=1234)
            info["build_seconds"] = time.time() - t
            m = metrics_for_target(info)
            m["distill_loss_beta0"] = float(obj.loss(prep, info, seed=1234).detach())
            m.update({"batch": bi, "domain": b["domain"], "suite": b["suite"],
                      "task_id": b["task_id"], "chunk": b["chunk"]})
            per_batch.append(m)
            # fixed-family K sweep: identity + noise seeds 1..7
            specs = build_views(seed=1234)
            obj_k = SGCpa(policy, a_basis, seed=1234,
                          views=[("identity", lambda im: im)] +
                                [(f"noise{i}", _noise_fn(0.05, 1000 + i)) for i in range(1, 8)])
            try:
                for K in (2, 4, 8):
                    sub = [(n, f) for n, f in obj_k.view_specs[:K]]
                    tmp = SGCpa(policy, a_basis, seed=1234, views=sub)
                    try:
                        inf = tmp.build_target(prep, seed=1234)
                        k_sweep[f"K{K}"].append(metrics_for_target(inf))
                    finally:
                        tmp.close()
            finally:
                obj_k.close()

        # K=1 actual-loss no-op
        obj1 = SGCpa(policy, a_basis, seed=1234, views=[("identity", lambda im: im)])
        try:
            prep = batches[0]["prepared"]
            inf = obj1.build_target(prep, seed=1234)
            loss = obj1.loss(prep, inf, seed=1234)
            obj1.adapter.beta.grad = None
            loss.backward()
            noop = {"loss": float(loss.detach()),
                    "grad_norm": float(obj1.adapter.beta.grad.norm())
                    if obj1.adapter.beta.grad is not None else 0.0}
        finally:
            obj1.close()

        # flow vs distill at K=4 (same teacher samples)
        prep = batches[0]["prepared"]
        inf = obj.build_target(prep, seed=1234)
        distill = float(obj.loss(prep, inf, seed=1234).detach())
        rep = policy.representation_extractor(prep["images"], prep["state"], prep["instruction"])
        gen = torch.Generator(device=DEV).manual_seed(7)
        flow_target = inf["target"][0] if inf["target"].dim() == 3 else inf["target"]
        flow = float(flow_matching_leftover_loss(policy.action_generator, rep, flow_target,
                                                 policy.config["action_horizon"], draws=8,
                                                 mode="self", rng=gen).detach())

        # Jacobian finite difference + A basis checks
        j = obj.jacobian_a(prep, seed=1234)
        sv = torch.linalg.svdvals(j)
        a_err = float((a_basis.T @ a_basis - torch.eye(5, dtype=torch.float64)).abs().max())
        proj = obj.nullspace(j)
        fd_errors = []
        d = torch.randn(j.shape[1], generator=torch.Generator().manual_seed(0), dtype=torch.float64)
        d = d / d.norm()
        vjp = j @ d
        for h in (1e-2, 1e-3):
            with torch.no_grad():
                obj.adapter.beta.copy_(h * d.float().to(DEV))
            zp = obj.write(prep, seed=1234).detach().double().cpu()
            with torch.no_grad():
                obj.adapter.beta.copy_(-h * d.float().to(DEV))
            zm = obj.write(prep, seed=1234).detach().double().cpu()
            with torch.no_grad():
                obj.adapter.beta.zero_()
            fd = (a_basis.T @ (zp - zm)) / (2 * h)
            fd_errors.append(float((fd - vjp).norm() / (vjp.norm() + 1e-12)))

        # cost breakdown
        t = time.time(); obj.build_target(prep, seed=1234); t_teacher = time.time() - t
        t = time.time(); obj.loss(prep, inf, seed=1234); t_loss = time.time() - t
        t = time.time(); obj.jacobian_a(prep, seed=1234); t_jac = time.time() - t
        t = time.time(); obj.nullspace(j); t_svd = time.time() - t
    finally:
        obj.close()

    agg = defaultdict(lambda: defaultdict(list))
    for m in per_batch:
        for k in ("D_view", "D_anchor_medoid", "D_anchor_mean_anchored_0.1",
                  "distill_loss_beta0"):
            agg[m["domain"]][k].append(m[k])
        medoid_view = list(m["groups"])[m["medoid_index"]]
        for g in ("xyz", "rot", "grip"):
            agg[m["domain"]][f"medoid_anchor_{g}"].append(m["groups"][medoid_view][g])
    medoid_counts = defaultdict(lambda: defaultdict(int))
    for m in per_batch:
        medoid_counts[m["domain"]][m["medoid_index"]] += 1
    summary = {d: {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in kv.items()}
               for d, kv in agg.items()}
    for d in summary:
        total = sum(medoid_counts[d].values())
        summary[d]["medoid_identity_frac"] = {"mean": medoid_counts[d].get(0, 0) / max(total, 1)}
        summary[d]["medoid_index_counts"] = dict(medoid_counts[d])
    k_agg = {K: {k: {"mean": float(np.mean([m[k] for m in ms])), "std": float(np.std([m[k] for m in ms]))}
                 for k in ("D_view", "D_anchor_medoid", "D_anchor_mean_anchored_0.1")}
             for K, ms in k_sweep.items()}
    results = {"n_batches": len(batches), "domains": sorted(summary),
               "target_summary": summary, "k_sweep_fixed_noise": k_agg,
               "k1_actual_loss_noop": noop,
               "flow_vs_distill": {"distill": distill, "flow": flow},
               "jacobian": {"shape": list(j.shape), "svals": [float(x) for x in sv],
                            "A_orthonormality_err": a_err,
                            "nullspace_relative": float((j @ proj).norm() / (proj.norm() + 1e-12)),
                            "fd_relative_errors": fd_errors},
               "cost_seconds": {"teacher": t_teacher, "loss": t_loss, "jacobian": t_jac, "svd": t_svd},
               "total_seconds": time.time() - t0}
    (args.out / "sg_cpa_4_0_final.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("domains", "k1_actual_loss_noop", "flow_vs_distill",
                                              "jacobian", "cost_seconds", "total_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
