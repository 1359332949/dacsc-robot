"""Phase-1.1 SALT port driver (CoRP-native flow convention, SALT B.4 gate).

modes: frozen | salt | self
gate-mode: feature (SALT B.4, requires --calibration) | always (diagnostic ablation)
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import (  # noqa: E402
    DUMMY_ACTION, MAX_STEPS, SETTLING_STEPS, configure_libero, is_success, policy_observation,
)
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.salt_tta import (  # noqa: E402
    FeatureChangeGate, SaltTTA, flow_matching_leftover_loss, probe_flow_loss,
)
from dacsc.sg_salt import sg_salt_loss  # noqa: E402
from dacsc.sg_cpa import SGCpa  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402

CLEAN_ROOT = Path("/hy-tmp/dacsc-robot-784fe2e/repos/CoRP/third_party/LIBERO/libero/libero")


def build_env(args, bddl):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                              render_gpu_device_id=args.render_gpu)


def observed(args, policy, env, obs, task):
    batch = policy_observation(obs, task.language, policy)
    return prepare_observation(batch, config=policy.config, normalizer=policy.normalizer,
                               device=args.device)


def sample_norm(policy, prepared, num_steps):
    with torch.no_grad():
        rep = policy.representation_extractor(**prepared)
        chunk = policy.action_generator.sample(rep, num_steps=num_steps)
    return rep, chunk


def beta_snapshot(tta) -> dict:
    b = tta.adapter.beta.detach().double().cpu()
    if b.numel() == 0:
        return {}
    q = torch.quantile(b, torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], dtype=torch.float64))
    return {
        "beta_l2": float(b.norm()),
        "beta_abs_mean": float(b.abs().mean()),
        "beta_abs_max": float(b.abs().max()),
        "frac_ge_095eta": float((b.abs() >= 0.95 * tta.adapter.eta).float().mean()),
        "frac_ge_099eta": float((b.abs() >= 0.99 * tta.adapter.eta).float().mean()),
        "neg_boundary_frac": float((b <= -0.99 * tta.adapter.eta).float().mean()),
        "pos_boundary_frac": float((b >= 0.99 * tta.adapter.eta).float().mean()),
        "quantiles": [float(x) for x in q],
    }


def run_episode(args, policy, env, task, task_id: int, seed, episode: int, tta, gate,
                a_basis, calibration_missing: bool):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    env.seed(seed); env.reset()
    from libero.libero import get_libero_path
    states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    states = torch.load(states_path, weights_only=False)
    obs = env.set_init_state(states[episode % len(states)])
    for _ in range(SETTLING_STEPS):
        obs, _, done, _ = env.step(DUMMY_ACTION)
        if done: break
    if tta is not None:
        if args.mode == "sg_cpa":
            tta.reset(seed=seed)
        else:
            tta.reset()
    if gate is not None:
        gate.reset()
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    draw_rng = torch.Generator(device=gen_device).manual_seed(seed + 17)
    success, done, steps = False, False, 0
    prev_norm = None
    updates = 0
    gate_step = -1
    predict_index = 0
    last_penalty = None
    last_salt = None
    sg_records = []
    sg_attempted = sg_accepted = sg_rejected = sg_skipped = 0
    replan_times = []
    diag = {"task_id": int(task_id), "init_state_index": int(episode),
            "seed": int(seed), "replans": [], "updates": [], "gate_mode": args.gate_mode,
            "mode": args.mode, "threshold": None if gate is None else gate.threshold}
    while not success and not done and steps < MAX_STEPS[args.suite]:
        prepared = observed(args, policy, env, obs, task)
        with torch.no_grad():
            rep = policy.representation_extractor(**prepared)
        gate_score = None
        gate_active = False
        if args.gate_mode == "always":
            gate_active = True
            if gate_step < 0: gate_step = predict_index
        elif gate is not None:
            gate_score, gate_active = gate.step(rep.tokens)
            if gate_active and gate_step < 0: gate_step = predict_index
        replan = {"index": predict_index, "score": gate_score, "active": gate_active,
                  "updates": 0, "probe_before": None, "probe_after": None}
        if tta is not None and gate_active and prev_norm is not None and args.mode == "sg_cpa":
            t_step = time.time()
            action_seed = seed + 7919 * predict_index
            rec = tta.step(prepared, seed=action_seed, view_seed=action_seed + 1)
            replan_times.append(time.time() - t_step)
            sg_records.append(rec)
            sg_attempted += 1
            if rec.get("skipped"):
                sg_skipped += 1
            elif rec.get("accepted"):
                sg_accepted += 1
            else:
                sg_rejected += 1
            updates += 1
            replan["updates"] += 1
            if args.diag_out is not None:
                diag["updates"].append({"replan": predict_index,
                                        "action_seed": int(action_seed),
                                        "view_seed": int(action_seed + 1),
                                        **rec, **beta_snapshot(tta)})
        elif tta is not None and gate_active and prev_norm is not None:
            if args.diag_out is not None:
                replan["probe_before"] = probe_flow_loss(
                    policy.action_generator, rep, prev_norm, args.exec_len,
                    mode="leftover" if args.mode == "salt" else "self")
            for _ in range(args.adapt_steps):
                if args.mode == "self":
                    with torch.no_grad():
                        target = policy.action_generator.sample(rep, num_steps=args.num_steps)[0].detach()
                    mode = "self"
                else:
                    target, mode = prev_norm, "leftover"
                if args.mode == "sg_salt":
                    loss, salt_loss, penalty = sg_salt_loss(
                        policy, tta, rep, target, args.exec_len, draws=args.draws,
                        mode=mode, rng=draw_rng, a_basis=a_basis,
                        lam=args.lambda_protect)
                else:
                    loss = flow_matching_leftover_loss(policy.action_generator, rep, target,
                                                       args.exec_len, draws=args.draws, mode=mode,
                                                       rng=draw_rng)
                    salt_loss, penalty = loss.detach(), None
                tta.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = tta.adapter.beta.grad
                grad_l2 = float(grad.norm()) if grad is not None else 0.0
                grad_max = float(grad.abs().max()) if grad is not None else 0.0
                before = tta.adapter.beta.detach().clone()
                tta.optimizer.step()
                with torch.no_grad():
                    tta.adapter.beta.clamp_(-args.eta, args.eta)
                delta = float((tta.adapter.beta.detach() - before).norm())
                updates += 1
                replan["updates"] += 1
                last_penalty = None if penalty is None else float(penalty.detach())
                last_salt = float(salt_loss.detach())
                if args.diag_out is not None:
                    diag["updates"].append({
                        "replan": predict_index, "step": replan["updates"] - 1,
                        "loss": float(loss.detach()), "salt_loss": float(salt_loss),
                        "penalty": None if penalty is None else float(penalty),
                        "grad_l2": grad_l2, "grad_max": grad_max,
                        "delta_beta_l2": delta, **beta_snapshot(tta),
                    })
            if args.diag_out is not None:
                replan["probe_after"] = probe_flow_loss(
                    policy.action_generator, rep, prev_norm, args.exec_len,
                    mode="leftover" if args.mode == "salt" else "self")
                replan["beta"] = beta_snapshot(tta)
        diag["replans"].append(replan)
        _, chunk = sample_norm(policy, prepared, args.num_steps)
        prev_norm = chunk[0].detach()
        actions = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
        predict_index += 1
        for a in actions[: args.exec_len]:
            obs, reward, done, info = env.step(np.clip(a, -1.0, 1.0))
            steps += 1
            success = is_success(env, info, reward, done)
            if success or done or steps >= MAX_STEPS[args.suite]: break
    summary = {}
    if tta is not None:
        summary = beta_snapshot(tta)

    diag["result"] = {"success": bool(success), "steps": steps, "updates": updates,
                      "gate_step": gate_step}
    if args.diag_out is not None:
        args.diag_out.mkdir(parents=True, exist_ok=True)
        path = args.diag_out / f"task{task_id}_{seed}_{args.mode}_{args.gate_mode}.json"
        path.write_text(json.dumps(diag))
    sg_summary = {}
    if sg_records:
        acc = [r for r in sg_records if not r.get("skipped")]
        sg_summary = {
            "retain_mean": float(np.mean([r["retained_fraction"] for r in acc])) if acc else None,
            "backtrack_rate": float(np.mean([r["backtracks"] > 0 for r in acc])) if acc else None,
            "accepted_rate": float(np.mean([r["accepted"] for r in acc])) if acc else None,
            "d_a_current_obs_total_last": acc[-1]["cumulative_a_change"] if acc else None,
            "d_a_local_last": acc[-1]["local_a_change"] if acc else None,
            "attempted": sg_attempted, "accepted": sg_accepted,
            "rejected": sg_rejected, "skipped": sg_skipped,
            "beta_abs_max_last": acc[-1]["beta_abs_max"] if acc else None,
            "replan_seconds_mean": float(np.mean(replan_times)) if replan_times else None,
        }
    return {"success": bool(success), "steps": steps, "updates": updates,
            "gate_step": gate_step, "beta": summary,
            "penalty_last": last_penalty, "salt_loss_last": last_salt,
            "init_state_index": int(episode), "sg": sg_summary}


def task_id_of(task) -> int:
    return int(getattr(task, "task_id", 0))
def calibrate(args, policy):
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()[args.suite]()
    gate = FeatureChangeGate(num_tokens=args.gate_tokens)
    per_task: dict[int, list[list[torch.Tensor]]] = {}
    seeds, frames, episodes = [], 0, 0
    for tid in [int(x) for x in args.calib_tasks.split(",")]:
        task = suite.get_task(tid)
        bddl = CLEAN_ROOT / "bddl_files" / task.problem_folder / f"{task.name}.bddl"
        per_task[tid] = []
        for ep in range(args.calib_episodes):
            env = build_env(args, bddl)
            seed = args.seed + tid * 1000 + ep
            np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            env.seed(seed); env.reset()
            states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                weights_only=False)
            obs = env.set_init_state(states[ep % len(states)])
            for _ in range(SETTLING_STEPS):
                obs, _, done, _ = env.step(DUMMY_ACTION)
                if done: break
            gate.reset()
            changes: list[torch.Tensor] = []
            steps = 0
            while steps < int(args.calib_steps):
                prepared = observed(args, policy, env, obs, task)
                with torch.no_grad():
                    rep = policy.representation_extractor(**prepared)
                change = gate.token_change(rep.tokens)
                if gate.prev is not None:
                    changes.append(change.detach().cpu().clone())
                _, chunk = sample_norm(policy, prepared, args.num_steps)
                actions = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
                for a in actions[: args.exec_len]:
                    obs, _, done, _ = env.step(np.clip(a, -1.0, 1.0))
                    steps += 1
                    if done or steps >= int(args.calib_steps): break
                if done: break
            env.close()
            per_task[tid].append(changes)
            frames += len(changes)
            episodes += 1
            seeds.append(seed)
            print(f"[calib] task={tid} ep={ep} transitions={len(changes)}", flush=True)
    stats = gate.calibrate(per_task)
    payload = {"statistics": stats, "gate_tokens": args.gate_tokens,
               "calib_tasks": [int(x) for x in args.calib_tasks.split(",")],
               "calib_episodes": args.calib_episodes, "seeds": seeds,
               "frames": frames, "episodes": episodes,
               "checkpoint": str(_CORP_REPO / "checkpoints/corp_libero.pt"),
               "suite": args.suite, "exec_len": args.exec_len, "num_steps": args.num_steps}
    args.calibration.write_text(json.dumps(payload))
    print(f"[calib] threshold={stats['threshold']:.4f} frames={frames} episodes={episodes} -> {args.calibration}",
          flush=True)


def main() -> int:
    from dacsc.upstream import verify_upstream
    upstream = verify_upstream(["CoRP"])
    print(f"[upstream] verified: {sorted(upstream)}", flush=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--mode", choices=["frozen", "salt", "self", "sg_salt", "sg_cpa"], default="frozen")
    ap.add_argument("--sgcpa-alpha", type=float, default=0.1)
    ap.add_argument("--sgcpa-sigma", type=float, default=0.05)
    ap.add_argument("--sgcpa-eta", type=float, default=0.05)
    ap.add_argument("--a-change-tol", type=float, default=5e-3)
    ap.add_argument("--sgcpa-aggregation", choices=["medoid", "mean_anchored"], default="medoid")
    ap.add_argument("--sgcpa-photo", type=float, default=1.05)
    ap.add_argument("--sgcpa-blur-sigma", type=float, default=0.5)
    ap.add_argument("--sgcpa-no-projection", action="store_true")
    ap.add_argument("--sgcpa-medoid-exclude-identity", action="store_true")
    ap.add_argument("--sgcpa-no-a-guard", action="store_true")
    ap.add_argument("--episodes-per-task", type=int, default=1)
    ap.add_argument("--lambda-protect", type=float, default=0.0)
    ap.add_argument("--a-basis", type=Path, default=None)
    ap.add_argument("--gate-mode", choices=["feature", "always"], default="feature")
    ap.add_argument("--gate-tokens", type=int, default=96, choices=[48, 96])
    ap.add_argument("--exec-len", type=int, default=6)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--adapt-steps", type=int, default=5)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--diag-out", type=Path, default=None,
                    help="directory for per-episode diagnostics JSON")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib-tasks", default="0,1,2,3,4")
    ap.add_argument("--calib-episodes", type=int, default=4)
    ap.add_argument("--calib-steps", default="120")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.calibrate and not args.tasks:
        ap.error("--tasks is required unless --calibrate")
    verify_release_checkpoint(_CORP_REPO / "checkpoints/corp_libero.pt")
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path

    policy = load_policy(_CORP_REPO / "checkpoints/corp_libero.pt", device=args.device)
    if args.calibrate:
        calibrate(args, policy)
        return 0
    gate = None
    if args.mode == "frozen":
        gate = None
    elif args.gate_mode == "feature":
        if args.calibration is None or not args.calibration.exists():
            raise SystemExit("--calibration is required for gate-mode=feature")
        blob = json.loads(args.calibration.read_text())
        gate = FeatureChangeGate(num_tokens=args.gate_tokens)
        gate.load(blob["statistics"] | {"n_tokens": blob["gate_tokens"]})
    if args.mode == "sg_cpa":
        if args.a_basis is None or not args.a_basis.exists():
            raise SystemExit("--a-basis is required for mode=sg_cpa")
        a_basis_t = torch.tensor(json.loads(args.a_basis.read_text())["action_subspace"],
                                 dtype=torch.float64)
        tta = SGCpa(policy, a_basis_t, alpha=args.sgcpa_alpha, sigma=args.sgcpa_sigma,
                    eta=args.sgcpa_eta, a_change_tol=args.a_change_tol,
                    seed=args.seed, aggregation=args.sgcpa_aggregation,
                    photo=args.sgcpa_photo, blur_sigma=args.sgcpa_blur_sigma,
                    use_projection=not args.sgcpa_no_projection,
                    a_guard=not args.sgcpa_no_a_guard,
                    medoid_include_identity=not args.sgcpa_medoid_exclude_identity)
    else:
        tta = SaltTTA(policy, eta=args.eta, adapt_steps=args.adapt_steps, draws=args.draws) \
            if args.mode in ("salt", "self", "sg_salt") else None
    a_basis = None
    a_basis_sha = None
    if args.mode in ("sg_salt", "sg_cpa"):
        if args.a_basis is None or not args.a_basis.exists():
            raise SystemExit("--a-basis is required for mode=sg_salt")
        import hashlib
        a_basis_sha = hashlib.sha256(args.a_basis.read_bytes()).hexdigest()
        a_basis = torch.tensor(json.loads(args.a_basis.read_text())["action_subspace"],
                               dtype=torch.float32)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for task_id in [int(x) for x in args.tasks.split(",")]:
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            for episode in range(args.episodes_per_task):
                env = build_env(args, bddl)
                seed = args.seed + task_id * 1000 + episode
                rec = run_episode(args, policy, env, task, task_id, seed, episode,
                                  tta, gate, a_basis, False)
                env.close()
                rec.update({"suite": args.suite, "task_id": task_id, "episode_id": episode,
                            "mode": args.mode, "gate_mode": args.gate_mode,
                            "gate_tokens": args.gate_tokens, "exec_len": args.exec_len,
                            "seed": seed, "lambda_protect": args.lambda_protect,
                            "a_basis_sha256": a_basis_sha})
                output.write(json.dumps(rec) + "\n"); output.flush()
                print(rec, flush=True)
    if tta is not None:
        tta.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
