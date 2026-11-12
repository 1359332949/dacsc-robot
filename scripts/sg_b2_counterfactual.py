"""B2 paired counterfactual branches: update vs no-update at the same state.

For each episode and branch replan t:
  * replay the frozen policy from the same init state to t (identical actions);
  * at t compute the SG-CPA gradient/J_A and candidate updates dbeta;
  * branch A (no-update, beta=0) and branch B (beta=dbeta) sample the action chunk
    with the SAME action latent seed, execute the full 12-step chunk, and record
    proximity progress at t and t+H;
  * tau = dp_update - dp_control.

Only proximity progress is used (geometric failed B0 acceptance).  No adaptation
during the branch; this is a simulator counterfactual, not a physical rollout.
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
CANDIDATES = ("proj_005", "full_005", "sens_005")


def fork():
    return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])


def sample(policy, rep, seed):
    with fork():
        torch.manual_seed(int(seed))
        with torch.no_grad():
            return policy.action_generator.sample(rep).detach()


def targets_for(instruction, suite):
    if suite == "libero_spatial":
        return [("akita_black_bowl_1", "plate_1")]
    instr = instruction.lower()
    for obj in ("alphabet_soup_1", "salad_dressing_1", "butter_1"):
        if obj.replace("_", " ") in instr or obj.split("_")[0] in instr:
            return [(obj, "basket_1")]
    raise RuntimeError(f"no target for {instruction}")


def resolve(names, raw):
    cand = raw + "_main"
    return names.index(cand) if cand in names else None


def proximity(r, pairs, names):
    bp = r["body_pos"]
    comps = []
    for obj, tgt in pairs:
        oi, ti = resolve(names, obj), resolve(names, tgt)
        if oi is None or ti is None:
            raise RuntimeError(f"unresolved {obj}/{tgt}")
        d = float(np.linalg.norm(np.array(bp[names[oi]]) - np.array(bp[names[ti]])))
        comps.append(d)
    return np.array(comps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/validation/b2"))
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--branches", type=int, default=3)   # replan indices 0..B-1
    ap.add_argument("--ethas", default="0.05")
    ap.add_argument("--candidates", default="proj_005,full_005,sens_005")
    ap.add_argument("--max-steps", type=int, default=240)
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
    cands = [c.strip() for c in args.candidates.split(",")]
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
                    # reference run: frozen policy to branch points (deterministic)
                    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                             render_gpu_device_id=0)
                    np.random.seed(seed); torch.manual_seed(seed); env.seed(seed); env.reset()
                    obs = env.set_init_state(states[ep % len(states)])
                    for _ in range(SETTLING_STEPS):
                        obs, _, done, _ = env.step(DUMMY_ACTION)
                        if done: break
                    prev_chunk = None
                    step = 0; branch = 0; done = False
                    while not done and branch < args.branches and step < args.max_steps:
                        raw = policy_observation(obs, task.language, policy)
                        prep = prepare_observation(raw, config=policy.config, normalizer=policy.normalizer,
                                                   device=torch.device(DEV))
                        rep = policy.representation_extractor(**prep)
                        action_seed = seed + 7919 * branch
                        with torch.no_grad():
                            sg.adapter.beta.zero_()
                        with fork():
                            torch.manual_seed(action_seed)
                            chunk_ctrl = policy.action_generator.sample(rep)
                        # signals + candidate dbeta from current beta=0
                        sg.adapter.beta.detach().zero_()
                        info = sg.build_target(prep, seed=action_seed, view_seed=action_seed + 1)
                        loss = sg.loss(prep, info, seed=action_seed)
                        sg.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        g = sg.adapter.beta.grad.detach().double().cpu().clone()
                        j = sg.jacobian_a(prep, seed=action_seed)
                        P = sg.nullspace(j)
                        zero = torch.zeros(768, dtype=torch.float64)
                        updates = {"zero": zero,
                                   "proj_005": -(0.05 * (P @ g)),
                                   "full_005": -(0.05 * g),
                                   "sens_005": -(0.05 * ((torch.eye(768, dtype=torch.float64) - P) @ g))}
                        views = sg.make_view_images(prep, action_seed + 1)
                        vacts = []
                        for img in views:
                            with fork():
                                torch.manual_seed(action_seed)
                                with torch.no_grad():
                                    r = policy.representation_extractor(img, prep["state"], prep["instruction"])
                                    vacts.append(policy.action_generator.sample(r))
                        pair_d = [float((vacts[i] - vacts[j]).abs().mean())
                                  for i in range(4) for j in range(i + 1, 4)]
                        hist = None
                        if prev_chunk is not None:
                            hist = float((prev_chunk[0, 6:] - chunk_ctrl[0, :6]).abs().mean())
                        positions = {n: [float(x) for x in env.sim.data.body_xpos[i]]
                                     for i, n in enumerate([str(x) for x in env.sim.model.body_names])}
                        names = list(positions)
                        pairs = targets_for(task.language, suite)
                        d0 = proximity({"body_pos": positions}, pairs, names)
                        # branch pairs
                        for cand in cands:
                            dbeta = updates[cand]
                            for mode in ("control", "update"):
                                env2 = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                                                          camera_widths=256, render_gpu_device_id=0)
                                np.random.seed(seed); torch.manual_seed(seed); env2.seed(seed); env2.reset()
                                o2 = env2.set_init_state(states[ep % len(states)])
                                for _ in range(SETTLING_STEPS):
                                    o2, _, d2, _ = env2.step(DUMMY_ACTION)
                                    if d2: break
                                # replay prefix with frozen policy (same actions as reference)
                                with torch.no_grad():
                                    sg.adapter.beta.zero_()
                                for tt in range(branch):
                                    raw2 = policy_observation(o2, task.language, policy)
                                    prep2 = prepare_observation(raw2, config=policy.config,
                                                                normalizer=policy.normalizer,
                                                                device=torch.device(DEV))
                                    r2 = policy.representation_extractor(**prep2)
                                    with fork():
                                        torch.manual_seed(seed + 7919 * tt)
                                        a2 = policy.action_generator.sample(r2)
                                    ctrl2 = policy.normalizer.unnormalize_action(a2)[0].float().cpu().numpy()
                                    for x in ctrl2[:6]:
                                        o2, _, d2, _ = env2.step(np.clip(x, -1.0, 1.0))
                                        if d2: break
                                # branch action
                                if mode == "update":
                                    with torch.no_grad():
                                        sg.adapter.beta.copy_(dbeta.float().to(sg.adapter.beta.device))
                                else:
                                    with torch.no_grad():
                                        sg.adapter.beta.zero_()
                                with fork():
                                    torch.manual_seed(action_seed)
                                    chunk_zero = policy.action_generator.sample(rep)
                                rawb = policy_observation(o2, task.language, policy)
                                prepb = prepare_observation(rawb, config=policy.config,
                                                            normalizer=policy.normalizer,
                                                            device=torch.device(DEV))
                                repb = policy.representation_extractor(**prepb)
                                with fork():
                                    torch.manual_seed(action_seed)
                                    ab = policy.action_generator.sample(repb)
                                ctrlb = policy.normalizer.unnormalize_action(ab)[0].float().cpu().numpy()
                                act_change = float((ab - chunk_zero).abs().mean())
                                pos_before = {n: [float(x) for x in env2.sim.data.body_xpos[i]]
                                              for i, n in enumerate([str(x) for x in env2.sim.model.body_names])}
                                pb0 = proximity({"body_pos": pos_before}, pairs, list(pos_before))
                                success_b = False; done_b = False; steps_b = 0
                                for x in ctrlb[:12]:
                                    o2, reward, done_b, info2 = env2.step(np.clip(x, -1.0, 1.0))
                                    steps_b += 1
                                    success_b = is_success(env2, info2, reward, done_b)
                                    if success_b or done_b: break
                                pos_after = {n: [float(x) for x in env2.sim.data.body_xpos[i]]
                                             for i, n in enumerate([str(x) for x in env2.sim.model.body_names])}
                                pb1 = proximity({"body_pos": pos_after}, pairs, list(pos_after))
                                env2.close()
                                rows.append({"domain": domain, "suite": suite, "task_id": tid,
                                             "episode": ep, "seed": seed, "branch": branch,
                                             "steps": step, "candidate": cand, "mode": mode,
                                             "p_before": float(np.mean(1 - pb0 / np.maximum(d0, 1e-9))),
                                             "p_after": float(np.mean(1 - pb1 / np.maximum(d0, 1e-9))),
                                             "dp": float(np.mean(1 - pb1 / np.maximum(d0, 1e-9))
                                                         - np.mean(1 - pb0 / np.maximum(d0, 1e-9))),
                                             "success": bool(success_b), "steps_branch": steps_b,
                                             "dbeta_norm": float(dbeta.norm()),
                                             "action_change_vs_control": act_change,
                                             "mv_signal": -float(np.mean(pair_d)),
                                             "hist_signal": None if hist is None else -hist})
                        # advance reference by one replan (frozen actions)
                        with fork():
                            torch.manual_seed(action_seed)
                            ctrl = policy.normalizer.unnormalize_action(chunk_ctrl)[0].float().cpu().numpy()
                        prev_chunk = chunk_ctrl.detach()
                        for x in ctrl[:6]:
                            obs, reward, done, info2 = env.step(np.clip(x, -1.0, 1.0))
                            step += 1
                            if is_success(env, info2, reward, done) or done: break
                        branch += 1
                    env.close()
                    print("episode", domain, suite, tid, ep, "branches", branch, flush=True)
    finally:
        sg.close()
    (args.out / "b2.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print("rows", len(rows), "seconds", time.time() - t0)


if __name__ == "__main__":
    main()
