"""B2 v3 protocol: two-env replay (bitwise deterministic) + 4x6-step production execution.

At branch replan t both environments have executed identical frozen actions (beta=0,
matched latent seeds).  For 4 segments of 6 steps: control keeps beta=0, update keeps
the candidate beta; each segment observes, samples with matched latent seed, executes
the first 6 actions.  Zero-update preflight must reproduce identical trajectories.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero, policy_observation, is_success  # noqa: E402
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.sg_cpa import SGCpa  # noqa: E402
from sg_cpa_phase4_final import CKPT, A_FILE  # noqa: E402

DEV = "cuda:0"
CELLS = {
    ("Camera", "libero_spatial"): [705, 620, 671],
    ("Background", "libero_spatial"): [11, 19, 18],
    ("Camera", "libero_object"): [653, 701, 901],
    ("Background", "libero_object"): [3, 8, 18],
}


def fork():
    return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])


def sample(policy, rep, seed, beta_sg):
    with torch.no_grad():
        beta_sg.adapter.beta.zero_() if beta_sg is not None else None
    with fork():
        torch.manual_seed(int(seed))
        with torch.no_grad():
            return policy.action_generator.sample(rep).detach()


def targets_for(instruction, suite):
    if suite == "libero_spatial":
        return [("akita_black_bowl_1", "plate_1")]
    for obj in ("alphabet_soup_1", "salad_dressing_1", "butter_1"):
        if obj.replace("_", " ") in instruction.lower() or obj.split("_")[0] in instruction.lower():
            return [(obj, "basket_1")]
    raise RuntimeError("no target")


def resolve(names, raw):
    c = raw + "_main"
    return names.index(c) if c in names else None


def upd_from_grad(g, P, eta):
    I = torch.eye(768, dtype=torch.float64)
    return {"proj": -(eta * (P @ g)), "full": -(eta * g), "sens": -(eta * ((I - P) @ g))}


def single_env(env_cls, bddl, states, seed):
    env = env_cls(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    np.random.seed(seed); torch.manual_seed(seed); env.seed(seed); env.reset()
    obs = env.set_init_state(states[0])
    for _ in range(SETTLING_STEPS):
        obs, _, done, _ = env.step(DUMMY_ACTION)
        if done: break
    return env, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/validation/b2v3_pre"))
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--branch", type=int, default=8)
    ap.add_argument("--etas", default="1,10,100")
    ap.add_argument("--mode", default="preflight")  # preflight only for now
    args = ap.parse_args()
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
    rows = []
    t0 = time.time()
    try:
        for (domain, suite), tids in CELLS.items():
            for tid in tids:
                task = suites[suite].get_task(tid)
                bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                    weights_only=False)
                for ep in range(args.episodes):
                    seed = 30 + tid * 1000 + ep
                    envA, obsA = single_env(OffScreenRenderEnv, bddl, states, seed)
                    envB, obsB = single_env(OffScreenRenderEnv, bddl, states, seed)
                    names = [str(x) for x in envA.sim.model.body_names]
                    pairs = targets_for(task.language, suite)
                    oi, ti = resolve(names, pairs[0][0]), resolve(names, pairs[0][1])
                    d0 = float(np.linalg.norm(np.array(envA.sim.data.body_xpos[oi]) - np.array(envA.sim.data.body_xpos[ti])))
                    # prefix replay with frozen policy, matched seeds, beta=0
                    for k in range(args.branch):
                        prep = prepare_observation(policy_observation(obsA, task.language, policy),
                                                   config=policy.config, normalizer=policy.normalizer, device=torch.device(DEV))
                        rep = policy.representation_extractor(**prep)
                        with torch.no_grad(): sg.adapter.beta.zero_()
                        with fork():
                            torch.manual_seed(seed + 7919 * k)
                            with torch.no_grad():
                                cA = policy.action_generator.sample(rep)
                        ctrl = policy.normalizer.unnormalize_action(cA)[0].float().cpu().numpy()
                        for a in ctrl[:6]:
                            obsA, rA, dA, iA = envA.step(np.clip(a, -1.0, 1.0))
                        prepB = prepare_observation(policy_observation(obsB, task.language, policy),
                                                    config=policy.config, normalizer=policy.normalizer, device=torch.device(DEV))
                        repB = policy.representation_extractor(**prepB)
                        with fork():
                            torch.manual_seed(seed + 7919 * k)
                            with torch.no_grad():
                                cB = policy.action_generator.sample(repB)
                        ctrlB = policy.normalizer.unnormalize_action(cB)[0].float().cpu().numpy()
                        for a in ctrlB[:6]:
                            obsB, rB, dB, iB = envB.step(np.clip(a, -1.0, 1.0))
                    # branch quantities from envA observation
                    prep = prepare_observation(policy_observation(obsA, task.language, policy),
                                               config=policy.config, normalizer=policy.normalizer, device=torch.device(DEV))
                    rep = policy.representation_extractor(**prep)
                    with torch.no_grad(): sg.adapter.beta.zero_()
                    info = sg.build_target(prep, seed=seed + 7919 * args.branch, view_seed=seed + 7919 * args.branch + 1)
                    loss = sg.loss(prep, info, seed=seed + 7919 * args.branch)
                    sg.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    g = sg.adapter.beta.grad.detach().double().cpu().clone()
                    j = sg.jacobian_a(prep, seed=seed + 7919 * args.branch)
                    P = sg.nullspace(j)
                    eef = np.array(envA.sim.data.body_xpos[names.index("gripper0_eef")])
                    objp = np.array(envA.sim.data.body_xpos[oi])
                    stage = {"eef_obj_dist": float(np.linalg.norm(eef - objp)),
                             "lift": float(objp[2] - np.array(envA.sim.data.body_xpos[oi])[2])}
                    # preflight: eta grid, zero candidate
                    cands = {"zero": torch.zeros(768, dtype=torch.float64)}
                    for eta in [float(x) for x in args.etas.split(",")]:
                        for k, v in upd_from_grad(g, P, eta).items():
                            cands[f"{k}_{eta}"] = v
                    for cand_name, dbeta in cands.items():
                        with torch.no_grad(): sg.adapter.beta.copy_(dbeta.float().to(sg.adapter.beta.device))
                        trajA, trajB, dpA, dpB = [], [], None, None
                        # control env A keeps beta=0; update env B keeps candidate beta
                        for k in range(4):
                            # env A control
                            with torch.no_grad(): sg.adapter.beta.zero_()
                            prepA = prepare_observation(policy_observation(obsA, task.language, policy),
                                                        config=policy.config, normalizer=policy.normalizer, device=torch.device(DEV))
                            repA = policy.representation_extractor(**prepA)
                            with fork():
                                torch.manual_seed(seed + 7919 * (args.branch + k))
                                with torch.no_grad():
                                    chA = policy.action_generator.sample(repA)
                            cA = policy.normalizer.unnormalize_action(chA)[0].float().cpu().numpy()
                            for a in cA[:6]:
                                obsA, rA, dA, iA = envA.step(np.clip(a, -1.0, 1.0))
                                trajA.append([float(x) for x in envA.sim.data.body_xpos[oi]])
                                if is_success(envA, iA, rA, dA) or dA: break
                            # env B update
                            with torch.no_grad(): sg.adapter.beta.copy_(dbeta.float().to(sg.adapter.beta.device))
                            prepB = prepare_observation(policy_observation(obsB, task.language, policy),
                                                        config=policy.config, normalizer=policy.normalizer, device=torch.device(DEV))
                            repB = policy.representation_extractor(**prepB)
                            with fork():
                                torch.manual_seed(seed + 7919 * (args.branch + k))
                                with torch.no_grad():
                                    chB = policy.action_generator.sample(repB)
                            cB = policy.normalizer.unnormalize_action(chB)[0].float().cpu().numpy()
                            for a in cB[:6]:
                                obsB, rB, dB, iB = envB.step(np.clip(a, -1.0, 1.0))
                                trajB.append([float(x) for x in envB.sim.data.body_xpos[oi]])
                                if is_success(envB, iB, rB, dB) or dB: break
                        # reset envs by recreating for next candidate
                        envA.close(); envB.close()
                        envA, obsA = single_env(OffScreenRenderEnv, bddl, states, seed)
                        envB, obsB = single_env(OffScreenRenderEnv, bddl, states, seed)
                        # replay prefix again (same frozen actions)
                        for k in range(args.branch):
                            prep = prepare_observation(policy_observation(obsA, task.language, policy),
                                                       config=policy.config, normalizer=policy.normalizer, device=torch.device(DEV))
                            rep = policy.representation_extractor(**prep)
                            with torch.no_grad(): sg.adapter.beta.zero_()
                            with fork():
                                torch.manual_seed(seed + 7919 * k)
                                with torch.no_grad():
                                    ch = policy.action_generator.sample(rep)
                            ctrl = policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                            for a in ctrl[:6]: obsA, _, dA, _ = envA.step(np.clip(a, -1.0, 1.0))
                            prepB = prepare_observation(policy_observation(obsB, task.language, policy),
                                                        config=policy.config, normalizer=policy.normalizer, device=torch.device(DEV))
                            repB = policy.representation_extractor(**prepB)
                            with fork():
                                torch.manual_seed(seed + 7919 * k)
                                with torch.no_grad():
                                    ch = policy.action_generator.sample(repB)
                            ctrlB = policy.normalizer.unnormalize_action(ch)[0].float().cpu().numpy()
                            for a in ctrlB[:6]: obsB, _, dB, _ = envB.step(np.clip(a, -1.0, 1.0))
                        pa = np.array(trajA); pb = np.array(trajB)
                        L = min(len(pa), len(pb))
                        maxdiff = float(np.abs(pa[:L] - pb[:L]).max()) if L else None
                        rows.append({"domain": domain, "suite": suite, "task_id": tid, "episode": ep,
                                     "branch": args.branch, "candidate": cand_name,
                                     "dbeta_norm": float(dbeta.norm()), "max_traj_diff_control_vs_update": maxdiff,
                                     "n_control_steps": len(trajA), "n_update_steps": len(trajB),
                                     "stage": stage})
                    envA.close(); envB.close()
                    print("preflight", domain, suite, tid, ep, flush=True)
    finally:
        sg.close()
    (args.out / "preflight.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print("rows", len(rows), "seconds", time.time() - t0)


if __name__ == "__main__":
    main()
