"""B2 v3 formal branch experiment (two-env replay, 4x6-step protocol).

Formal execution is gated behind --authorize-formal.  Without it the script exits.
Candidate families:
  raw:  proj/full/sens at eta=100 (direction-mechanism, adapter coordinate clamp recorded)
  prod: SGCpa.step() (beta bound + A guard + backtrack), one accepted update
Termination/censoring: tau is only defined when BOTH branches execute the full 24
steps without success/termination; otherwise record censor_reason and skip.
Action cap (pre-registered): mean |delta action| > 0.05 -> censor and record.
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
CELLS = {("Camera", "libero_spatial"): [705, 620, 671], ("Background", "libero_spatial"): [11, 19, 18],
         ("Camera", "libero_object"): [653, 701, 901], ("Background", "libero_object"): [3, 8, 18]}
BRANCHES = (8, 12)
ACTION_CAP = 0.05
DELTA = 0.01


def fork():
    return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])


def target_name(suite, instruction):
    if suite == "libero_spatial":
        return "akita_black_bowl_1", "plate_1"
    for obj in ("alphabet_soup_1", "salad_dressing_1", "butter_1"):
        if obj.split("_")[0] in instruction.lower():
            return obj, "basket_1"
    raise RuntimeError("no target")


def make_env(env_cls, bddl, states, seed):
    env = env_cls(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    np.random.seed(seed); torch.manual_seed(seed); env.seed(seed); env.reset()
    obs = env.set_init_state(states[0])
    for _ in range(SETTLING_STEPS):
        obs, _, done, _ = env.step(DUMMY_ACTION)
        if done: break
    return env, obs


def prep_of(policy, obs, task):
    return prepare_observation(policy_observation(obs, task.language, policy),
                               config=policy.config, normalizer=policy.normalizer, device=torch.device(DEV))


def chunk_of(policy, sg, prep, seed, beta=None):
    with torch.no_grad():
        sg.adapter.beta.zero_() if beta is None else sg.adapter.beta.copy_(beta)
    with fork():
        torch.manual_seed(seed)
        with torch.no_grad():
            return policy.action_generator.sample(prep["representation"] if "representation" in prep else policy.representation_extractor(**prep)).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/validation/b2v3"))
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--authorize-formal", action="store_true")
    args = ap.parse_args()
    if not args.authorize_formal:
        raise SystemExit("formal B2 is not authorized; run preflight audits instead")
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
    rows = []
    try:
        for (domain, suite), tids in CELLS.items():
            for tid in tids:
                task = suites[suite].get_task(tid)
                bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                    weights_only=False)
                for ep in range(args.episodes):
                    seed = 30 + tid * 1000 + ep
                    envA, oA = make_env(OffScreenRenderEnv, bddl, states, seed)
                    envB, oB = make_env(OffScreenRenderEnv, bddl, states, seed)
                    names = [str(x) for x in envA.sim.model.body_names]
                    obj, tgt = target_name(suite, task.language)
                    oi, ti = names.index(obj + "_main"), names.index(tgt + "_main")
                    obj0 = np.array(envA.sim.data.body_xpos[oi])
                    step = 0; branch = 0; done = False
                    while not done and branch <= max(BRANCHES) and step < 240:
                        if branch in BRANCHES:
                            preA = prep_of(policy, oA, task); preB = prep_of(policy, oB, task)
                            eef = np.array(envA.sim.data.body_xpos[names.index("gripper0_eef")])
                            op = np.array(envA.sim.data.body_xpos[oi])
                            contact = float(np.linalg.norm(eef - op)); lift = float(op[2] - obj0[2])
                            if contact > 0.10:
                                pass  # skip branch, advance below
                            else:
                                # candidate updates at branch
                                sg.adapter.beta.zero_()
                                target = sg.build_target(preA, seed=seed + 7919 * branch, view_seed=seed + 7919 * branch + 1)
                                loss = sg.loss(preA, target, seed=seed + 7919 * branch)
                                sg.optimizer.zero_grad(set_to_none=True); loss.backward()
                                g = sg.adapter.beta.grad.detach().double().cpu().clone()
                                j = sg.jacobian_a(preA, seed=seed + 7919 * branch); P = sg.nullspace(j)
                                I = torch.eye(768, dtype=torch.float64)
                                raw = {"proj": -(100.0 * (P @ g)), "full": -(100.0 * g),
                                       "sens": -(100.0 * ((I - P) @ g))}
                                # production candidate via step()
                                sg.adapter.beta.zero_()
                                rec = sg.step(preA, seed=seed + 7919 * branch,
                                              view_seed=seed + 7919 * branch + 1, target_info=target)
                                prod_beta = sg.adapter.beta.detach().clone()
                                sg.adapter.beta.zero_()
                                candidates = {}
                                for k, v in raw.items():
                                    candidates[f"raw_{k}_100"] = v.float()
                                candidates["prod_step_005"] = prod_beta.float()
                                for cname, dbeta in candidates.items():
                                    eff = dbeta.double().clamp(-0.15, 0.15)
                                    clip_ratio = float((dbeta.double().abs() > 0.15).float().mean())
                                    # control / update chunks from same observation + latent
                                    controls, updates = [], []
                                    for k in range(4):
                                        for env, pre, store in ((envA, preA, controls), (envB, preB, updates)):
                                            with torch.no_grad():
                                                sg.adapter.beta.zero_()
                                            seed_k = seed + 7919 * (branch + k)
                                            ch_ctrl = chunk_of(policy, sg, pre, seed_k)
                                            with torch.no_grad():
                                                sg.adapter.beta.copy_(dbeta.to(sg.adapter.beta.device))
                                            ch_upd = chunk_of(policy, sg, pre, seed_k)
                                            store.append((ch_ctrl, ch_upd))
                                    ac = float((updates[0][1] - updates[0][0]).abs().mean())
                                    if ac > ACTION_CAP:
                                        rows.append({"cand": cname, "censor": "action_cap", "action_change": ac})
                                        continue
                                    # execute 4x6 for control (A) and update (B)
                                    trA, trB, stepsA, stepsB = [], [], 0, 0
                                    succA = succB = False; ok = True
                                    for k in range(4):
                                        for env, obs, tr, isA, chunk_idx in (
                                                (envA, oA, trA, True, 0), (envB, oB, trB, False, 1)):
                                            pre = prep_of(policy, obs, task)
                                            with torch.no_grad():
                                                sg.adapter.beta.zero_() if isA else sg.adapter.beta.copy_(dbeta.to(sg.adapter.beta.device))
                                            ch = chunk_of(policy, sg, pre, seed + 7919 * (branch + k))
                                            ctrl = policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                                            for a in ctrl[:6]:
                                                obs, reward, d, info = env.step(np.clip(a, -1.0, 1.0))
                                                tr.append([float(x) for x in env.sim.data.body_xpos[oi]])
                                                if isA: stepsA += 1; succA = succA or is_success(env, info, reward, d)
                                                else: stepsB += 1; succB = succB or is_success(env, info, reward, d)
                                                if d: break
                                            if isA: oA = obs
                                            else: oB = obs
                                        if succA or succB: ok = False; break
                                    if not ok or stepsA != stepsB:
                                        rows.append({"cand": cname, "censor": "termination_or_step_mismatch",
                                                     "stepsA": stepsA, "stepsB": stepsB,
                                                     "succA": succA, "succB": succB})
                                    else:
                                        d0 = float(np.linalg.norm(np.array(envA.sim.data.body_xpos[oi]) - np.array(envA.sim.data.body_xpos[ti])))
                                        dpA = float(1 - np.linalg.norm(np.array(trA[-1]) - np.array(envA.sim.data.body_xpos[ti])) / max(d0, 1e-9))
                                        dpB = float(1 - np.linalg.norm(np.array(trB[-1]) - np.array(envA.sim.data.body_xpos[ti])) / max(d0, 1e-9))
                                        rows.append({"domain": domain, "suite": suite, "task_id": tid, "episode": ep,
                                                     "branch": branch, "cand": cname, "tau_prox": dpB - dpA,
                                                     "dpA": dpA, "dpB": dpB, "action_change": ac,
                                                     "raw_norm": float(dbeta.double().norm()),
                                                     "effective_norm": float(eff.norm()),
                                                     "clip_ratio": clip_ratio,
                                                     "accepted": bool(rec.get("accepted", True)) if cname.startswith("prod") else True,
                                                     "backtracks": int(rec.get("backtracks", 0)) if cname.startswith("prod") else 0})
                                    envA.close(); envB.close()
                                    envA, oA = make_env(OffScreenRenderEnv, bddl, states, seed)
                                    envB, oB = make_env(OffScreenRenderEnv, bddl, states, seed)
                            # advance one replan frozen (both)
                        pre = prep_of(policy, oA, task)
                        ch = chunk_of(policy, sg, pre, seed + 7919 * branch)
                        ctrl = policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                        for a in ctrl[:6]:
                            oA, _, done, _ = envA.step(np.clip(a, -1.0, 1.0)); step += 1
                            if done: break
                        preB = prep_of(policy, oB, task)
                        chB = chunk_of(policy, sg, preB, seed + 7919 * branch)
                        ctrlB = policy.normalizer.unnormalize_action(chB)[0].float().cpu().numpy()
                        for a in ctrlB[:6]:
                            oB, _, doneB, _ = envB.step(np.clip(a, -1.0, 1.0))
                            if doneB: break
                        branch += 1
                    envA.close(); envB.close()
                    print("formal", domain, suite, tid, ep, flush=True)
    finally:
        sg.close()
    (args.out / "b2_v3.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print("rows", len(rows))


if __name__ == "__main__":
    main()
