"""B2 v3 formal branch experiment (corrected implementation).

Gates:
  --authorize-formal : required for the full 24-episode experiment
  --dry-run          : 1 task / 1 episode / 1 branch / 1 candidate only
Without either flag the script exits.

Protocol: at branch replan t (8 or 12) both environments have replayed identical
frozen-policy prefixes from the same init state. For each candidate the environments
are rebuilt and the prefix is replayed again; control keeps beta=0, update keeps the
candidate beta for 4 segments of 6 steps.  tau is only defined when both branches
execute exactly 24 steps without success/termination; otherwise censor.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero, policy_observation, is_success  # noqa: E402
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.sg_cpa import SGCpa  # noqa: E402
from sg_cpa_phase4_final import CKPT, A_FILE  # noqa: E402

DEV = "cuda:0"
ALL_CELLS = {("Camera", "libero_spatial"): [705, 620, 671], ("Background", "libero_spatial"): [11, 19, 18],
             ("Camera", "libero_object"): [653, 701, 901], ("Background", "libero_object"): [3, 8, 18]}
BRANCHES = (8, 12)
ACTION_CAP = 0.05
DELTA = 0.01
H = 4          # segments
SEG = 6        # env steps per replan
ZERO = lambda: torch.zeros(768, dtype=torch.float64)


def fork():
    return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])


def target_pair(suite, instruction):
    if suite == "libero_spatial":
        return "akita_black_bowl_1", "plate_1"
    for obj in ("alphabet_soup_1", "salad_dressing_1", "butter_1"):
        if obj.split("_")[0] in instruction.lower():
            return obj, "basket_1"
    raise RuntimeError("no target")


def make_env(env_cls, bddl, states, seed, episode=0):
    env = env_cls(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    np.random.seed(seed); torch.manual_seed(seed); env.seed(seed); env.reset()
    obs = env.set_init_state(states[episode % len(states)])
    for _ in range(SETTLING_STEPS):
        obs, _, done, _ = env.step(DUMMY_ACTION)
        if done: break
    return env, obs


def prep_of(policy, obs, task):
    return prepare_observation(policy_observation(obs, task.language, policy),
                               config=policy.config, normalizer=policy.normalizer, device=torch.device(DEV))


def sample_chunk(policy, sg, prep, seed, beta):
    """beta is REQUIRED and explicitly applied; no implicit zeroing."""
    with torch.no_grad():
        sg.adapter.beta.copy_(beta.to(sg.adapter.beta.device))
    rep = policy.representation_extractor(**prep)
    with fork():
        torch.manual_seed(int(seed))
        with torch.no_grad():
            return policy.action_generator.sample(rep).detach()


def uniform_scale(dbeta, bound=0.15):
    """Direction-preserving beta bound: scale the whole vector, never clamp coordinates."""
    inf = float(dbeta.double().abs().max()) if dbeta.numel() else 0.0
    alpha = 1.0 if inf <= bound else bound / max(inf, 1e-12)
    return (alpha * dbeta.double()), alpha, inf


def dist(env, names, oi, ti):
    return float(np.linalg.norm(np.array(env.sim.data.body_xpos[oi]) - np.array(env.sim.data.body_xpos[ti])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/validation/b2v3"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--branch", type=int, default=None)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--authorize-formal", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.authorize_formal):
        raise SystemExit("need --dry-run or --authorize-formal")
    if args.dry_run and (args.task_id is None or args.branch is None or args.candidate is None):
        raise SystemExit("dry-run requires --task-id --branch --candidate")
    if args.dry_run and str(args.out) == "/hy-tmp/validation/b2v3":
        raise SystemExit("dry-run requires an explicit independent --out directory")
    args.out.mkdir(parents=True, exist_ok=True)
    class C: pass
    c = C(); c.libero_src = Path("/hy-tmp/libero_plus"); c.libero_config_path = Path("/hy-tmp/libero_plus_config")
    c.render_gpu = 0; c.device = DEV
    configure_libero(c)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    policy = load_policy(CKPT, device=DEV)
    a_basis = torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float64)
    sg = SGCpa(policy, a_basis, seed=1234)
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in ("libero_spatial", "libero_object")}
    cells = ALL_CELLS if not args.dry_run else {
        k: [v for v in vals if v == args.task_id] for k, vals in ALL_CELLS.items()}
    cells = {k: v for k, v in cells.items() if v}
    rows = []
    try:
        for (domain, suite), tids in cells.items():
            for tid in tids:
                task = suites[suite].get_task(tid)
                bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                    weights_only=False)
                for ep in ([args.episode] if args.dry_run else range(2)):
                    seed = 30 + tid * 1000 + ep
                    branches = [args.branch] if args.dry_run else list(BRANCHES)
                    for branch in branches:
                        # rebuild envs for this branch, record d_ref at episode start
                        env0, obs0 = make_env(OffScreenRenderEnv, bddl, states, seed, ep)
                        names = [str(x) for x in env0.sim.model.body_names]
                        obj, tgt = target_pair(suite, task.language)
                        oi, ti = names.index(obj + "_main"), names.index(tgt + "_main")
                        names = [str(x) for x in env0.sim.model.body_names]
                        d_ref = dist(env0, names, oi, ti)
                        # replay prefix on a reference pair to reach the branch (and check eligibility)
                        envA, oA = make_env(OffScreenRenderEnv, bddl, states, seed, ep)
                        envB, oB = make_env(OffScreenRenderEnv, bddl, states, seed, ep)
                        steps = 0; succ = False; done = False; t = 0
                        while t < branch and not done:
                            for env, obs, isA in ((envA, oA, True), (envB, oB, False)):
                                prep = prep_of(policy, obs, task)
                                ch = sample_chunk(policy, sg, prep, seed + 7919 * t, ZERO())
                                ctrl = policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                                for a in ctrl[:SEG]:
                                    obs, reward, d, info = env.step(np.clip(a, -1.0, 1.0)); steps += 1
                                    succ = succ or is_success(env, info, reward, d); done = done or d
                                    if d: break
                                if isA: oA = obs
                                else: oB = obs
                            t += 1
                        eef = np.array(envA.sim.data.body_xpos[names.index("gripper0_eef")])
                        op = np.array(envA.sim.data.body_xpos[oi])
                        contact = float(np.linalg.norm(eef - op))
                        eligible = (contact <= 0.10) and (not succ) and (not done)
                        d_branch = dist(envA, names, oi, ti)
                        envA.close(); envB.close(); env0.close()
                        if not eligible:
                            rows.append({"domain": domain, "suite": suite, "task_id": tid, "episode": ep,
                                         "branch": branch, "censor": "stage_or_success",
                                         "contact": contact, "success_before": succ, "done_before": done})
                            continue
                        # candidates computed once at the branch observation
                        envT, oT = make_env(OffScreenRenderEnv, bddl, states, seed, ep)
                        # replay again on T to branch
                        for t in range(branch):
                            prep = prep_of(policy, oT, task)
                            ch = sample_chunk(policy, sg, prep, seed + 7919 * t, ZERO())
                            ctrl = policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                            for a in ctrl[:SEG]:
                                oT, _, d, _ = envT.step(np.clip(a, -1.0, 1.0))
                                if d: break
                        preB = prep_of(policy, oT, task)
                        with torch.no_grad(): sg.adapter.beta.zero_()
                        tgt_info = sg.build_target(preB, seed=seed + 7919 * branch, view_seed=seed + 7919 * branch + 1)
                        loss = sg.loss(preB, tgt_info, seed=seed + 7919 * branch)
                        sg.optimizer.zero_grad(set_to_none=True); loss.backward()
                        g = sg.adapter.beta.grad.detach().double().cpu().clone()
                        j = sg.jacobian_a(preB, seed=seed + 7919 * branch); P = sg.nullspace(j)
                        I = torch.eye(768, dtype=torch.float64)
                        raw = {"proj": -(100.0 * (P @ g)), "full": -(100.0 * g), "sens": -(100.0 * ((I - P) @ g))}
                        with torch.no_grad(): sg.adapter.beta.zero_()
                        rec = sg.step(preB, seed=seed + 7919 * branch, view_seed=seed + 7919 * branch + 1,
                                      target_info=tgt_info)
                        prod_beta = sg.adapter.beta.detach().clone()
                        with torch.no_grad(): sg.adapter.beta.zero_()
                        envT.close()
                        candidates = {}
                        raw_meta = {}
                        for k, v in raw.items():
                            applied, alpha, inf = uniform_scale(v)
                            candidates[f"raw_{k}_100"] = applied.float()
                            raw_meta[f"raw_{k}_100"] = {"alpha": alpha, "proposal_norm": float(v.double().norm()),
                                                        "applied_norm": float(applied.norm()),
                                                        "proposal_inf": inf,
                                                        "predicted_a_change": float((j @ applied).norm())}
                        prod_applied, _, _ = uniform_scale(prod_beta.double())
                        candidates["prod_step_005"] = prod_applied.float()
                        raw_meta["prod_step_005"] = None
                        if args.dry_run:
                            candidates = {args.candidate: candidates[args.candidate]}
                        for cname, dbeta in candidates.items():
                            # rebuild both envs for EVERY candidate and replay prefix
                            envA, oA = make_env(OffScreenRenderEnv, bddl, states, seed, ep)
                            envB, oB = make_env(OffScreenRenderEnv, bddl, states, seed, ep)
                            for t in range(branch):
                                for env, obs, isA in ((envA, oA, True), (envB, oB, False)):
                                    prep = prep_of(policy, obs, task)
                                    ch = sample_chunk(policy, sg, prep, seed + 7919 * t, ZERO())
                                    ctrl = policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                                    for a in ctrl[:SEG]:
                                        obs, _, d, _ = env.step(np.clip(a, -1.0, 1.0))
                                        if d: break
                                    if isA: oA = obs
                                    else: oB = obs
                            d_branch_A = dist(envA, names, oi, ti)
                            d_branch_B = dist(envB, names, oi, ti)

                            stepsA = stepsB = 0; succA = succB = False; doneA = doneB = False
                            ok = True; action_changes = []; censor = None
                            for k in range(H):
                                for env, obs, isA, beta in ((envA, oA, True, ZERO()), (envB, oB, False, dbeta)):
                                    prep = prep_of(policy, obs, task)
                                    if isA:
                                        ch1 = sample_chunk(policy, sg, prep, seed + 7919 * (branch + k), beta)
                                    else:
                                        ch0 = sample_chunk(policy, sg, prep, seed + 7919 * (branch + k), ZERO())
                                        ch1 = sample_chunk(policy, sg, prep, seed + 7919 * (branch + k), beta)
                                        da = float((ch1 - ch0).abs().mean())
                                        action_changes.append(da)
                                        if da > ACTION_CAP:
                                            censor = "action_cap"; ok = False; break
                                    ctrl = policy.normalizer.unnormalize_action(ch1)[0].float().cpu().numpy()
                                    for a in ctrl[:SEG]:
                                        obs, reward, d, info = env.step(np.clip(a, -1.0, 1.0))
                                        if isA:
                                            stepsA += 1; succA = succA or is_success(env, info, reward, d); doneA = doneA or d
                                        else:
                                            stepsB += 1; succB = succB or is_success(env, info, reward, d); doneB = doneB or d
                                        if d: break
                                    if isA: oA = obs
                                    else: oB = obs
                                if not ok: break
                                if doneA or doneB or succA or succB: ok = False; censor = censor or "termination_or_success"; break
                            complete = (ok and stepsA == 24 and stepsB == 24 and not succA and not succB
                                        and not doneA and not doneB)
                            if complete:
                                d_end_A = dist(envA, names, oi, ti); d_end_B = dist(envB, names, oi, ti)
                                dpA = (d_branch_A - d_end_A) / d_ref
                                dpB = (d_branch_B - d_end_B) / d_ref
                                rows.append({"domain": domain, "suite": suite, "task_id": tid, "episode": ep,
                                             "branch": branch, "candidate": cname,
                                             "target_pair": [obj, tgt], "target_body_ids": [oi, ti],
                                             "d_ref": d_ref, "d_branch_control": d_branch_A, "d_branch_update": d_branch_B,
                                             "d_end_control": d_end_A, "d_end_update": d_end_B,
                                             "dp_control": dpA, "dp_update": dpB, "tau_prox": dpB - dpA,
                                             "steps_control": stepsA, "steps_update": stepsB,
                                             "action_changes": action_changes, "action_change": float(np.mean(action_changes)),
                                             "proposal_norm": (raw_meta[cname]["proposal_norm"] if raw_meta.get(cname) else rec.get("raw_d_beta_norm")),
                                             "applied_norm": (raw_meta[cname]["applied_norm"] if raw_meta.get(cname) else float(prod_applied.norm())),
                                             "last_attempt_norm": (None if raw_meta.get(cname) else rec.get("d_beta_norm")),
                                             "alpha": (raw_meta[cname]["alpha"] if raw_meta.get(cname) else None),
                                             "clip_ratio": float((dbeta.double().abs() > 0.15).float().mean()),
                                             "beta_max": float(dbeta.double().abs().max()),
                                             "predicted_a_change": (raw_meta[cname]["predicted_a_change"] if raw_meta.get(cname) else float((j @ dbeta.double()).norm())),
                                             "step_scale": rec.get("step_scale") if cname.startswith("prod") else None,
                                             "accepted": rec.get("accepted") if cname.startswith("prod") else None,
                                             "backtracks": rec.get("backtracks") if cname.startswith("prod") else None,
                                             "local_a_change": rec.get("local_a_change") if cname.startswith("prod") else None,
                                             "cumulative_a_change": rec.get("cumulative_a_change") if cname.startswith("prod") else None,
                                             "target_contact": contact})
                            else:
                                rows.append({"domain": domain, "suite": suite, "task_id": tid, "episode": ep,
                                             "branch": branch, "candidate": cname, "censor": censor or "incomplete",
                                             "steps_control": stepsA, "steps_update": stepsB,
                                             "success_control": bool(succA), "success_update": bool(succB),
                                             "done_control": bool(doneA), "done_update": bool(doneB),
                                             "action_changes": action_changes})
                            envA.close(); envB.close()
                    print("formal", domain, suite, tid, ep, flush=True)
    finally:
        sg.close()
    (args.out / "b2_v3.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print("rows", len(rows))


if __name__ == "__main__":
    main()
