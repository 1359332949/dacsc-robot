"""B2 v2: single-env state save/restore counterfactual branches (update vs control).

At a branch state S: compute control/update action chunks from the SAME observation
and latent seed, then for each candidate/mode restore S, execute the 12-step chunk,
record proximity progress.  This removes cross-env nondeterminism.
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


def targets_for(instruction, suite):
    if suite == "libero_spatial":
        return [("akita_black_bowl_1", "plate_1")]
    instr = instruction.lower()
    for obj in ("alphabet_soup_1", "salad_dressing_1", "butter_1"):
        if obj.replace("_", " ") in instr or obj.split("_")[0] in instr:
            return [(obj, "basket_1")]
    raise RuntimeError(f"no target for {instruction}")


def resolve(names, raw):
    c = raw + "_main"
    return names.index(c) if c in names else None


def prox(pos, names, pairs):
    comps = []
    for obj, tgt in pairs:
        oi, ti = resolve(names, obj), resolve(names, tgt)
        if oi is None or ti is None:
            raise RuntimeError(f"unresolved {obj}/{tgt}")
        comps.append(float(np.linalg.norm(np.array(pos[names[oi]]) - np.array(pos[names[ti]]))))
    return np.array(comps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/validation/b2v2"))
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--branches", default="2,4")
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
    cands = [x.strip() for x in args.candidates.split(",")]
    branches = [int(x) for x in args.branches.split(",")]
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
                    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                             render_gpu_device_id=0)
                    np.random.seed(seed); torch.manual_seed(seed); env.seed(seed); env.reset()
                    obs = env.set_init_state(states[ep % len(states)])
                    for _ in range(SETTLING_STEPS):
                        obs, _, done, _ = env.step(DUMMY_ACTION)
                        if done: break
                    step = 0; branch = 0; done = False; prev_chunk = None
                    while not done and branch < max(branches) + 1 and step < args.max_steps:
                        if branch in branches:
                            raw = policy_observation(obs, task.language, policy)
                            prep = prepare_observation(raw, config=policy.config, normalizer=policy.normalizer,
                                                       device=torch.device(DEV))
                            rep = policy.representation_extractor(**prep)
                            action_seed = seed + 7919 * branch
                            sg.adapter.beta.detach().zero_()
                            info = sg.build_target(prep, seed=action_seed, view_seed=action_seed + 1)
                            loss = sg.loss(prep, info, seed=action_seed)
                            sg.optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            g = sg.adapter.beta.grad.detach().double().cpu().clone()
                            j = sg.jacobian_a(prep, seed=action_seed)
                            P = sg.nullspace(j)
                            updates = {"zero": torch.zeros(768, dtype=torch.float64),
                                       "proj_005": -(0.05 * (P @ g)),
                                       "full_005": -(0.05 * g),
                                       "sens_005": -(0.05 * ((torch.eye(768, dtype=torch.float64) - P) @ g))}
                            views = sg.make_view_images(prep, action_seed + 1)
                            va = []
                            for img in views:
                                with fork():
                                    torch.manual_seed(action_seed)
                                    with torch.no_grad():
                                        r = policy.representation_extractor(img, prep["state"], prep["instruction"])
                                        va.append(policy.action_generator.sample(r))
                            mv = -float(np.mean([float((va[i] - va[j]).abs().mean())
                                                 for i in range(4) for j in range(i + 1, 4)]))
                            names = [str(x) for x in env.sim.model.body_names]
                            pairs = targets_for(task.language, suite)
                            pos0 = {n: [float(x) for x in env.sim.data.body_xpos[i]] for i, n in enumerate(names)}
                            d0 = prox(pos0, names, pairs)
                            state = env.sim.get_state()
                            # chunks sampled from the same observation/latent
                            with fork():
                                torch.manual_seed(action_seed)
                                with torch.no_grad():
                                    sg.adapter.beta.zero_()
                                    ctrl_chunk = policy.action_generator.sample(rep)
                            hist = None
                            if prev_chunk is not None:
                                hist = -float((prev_chunk[0, 6:] - ctrl_chunk[0, :6]).abs().mean())
                            for cand in cands:
                                with torch.no_grad():
                                    sg.adapter.beta.copy_(updates[cand].float().to(sg.adapter.beta.device))
                                with fork():
                                    torch.manual_seed(action_seed)
                                    with torch.no_grad():
                                        upd_chunk = policy.action_generator.sample(rep)
                                with torch.no_grad():
                                    sg.adapter.beta.zero_()
                                act_change = float((upd_chunk - ctrl_chunk).abs().mean())
                                for mode, chunk in (("control", ctrl_chunk), ("update", upd_chunk)):
                                    env.sim.set_state(state); env.sim.forward()
                                    posb = {n: [float(x) for x in env.sim.data.body_xpos[i]]
                                            for i, n in enumerate(names)}
                                    ctrl = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
                                    success_b = False; done_b = False; steps_b = 0
                                    for a in ctrl[:12]:
                                        _, reward, done_b, info2 = env.step(np.clip(a, -1.0, 1.0))
                                        steps_b += 1
                                        success_b = is_success(env, info2, reward, done_b)
                                        if success_b or done_b: break
                                    posa = {n: [float(x) for x in env.sim.data.body_xpos[i]]
                                            for i, n in enumerate(names)}
                                    pa0 = prox(posb, names, pairs); pa1 = prox(posa, names, pairs)
                                    p_before = float(np.mean(1 - pa0 / np.maximum(d0, 1e-9)))
                                    p_after = float(np.mean(1 - pa1 / np.maximum(d0, 1e-9)))
                                    rows.append({"domain": domain, "suite": suite, "task_id": tid,
                                                 "episode": ep, "seed": seed, "branch": branch,
                                                 "candidate": cand, "mode": mode,
                                                 "p_before": p_before, "p_after": p_after,
                                                 "dp": p_after - p_before, "success": bool(success_b),
                                                 "steps_branch": steps_b,
                                                 "dbeta_norm": float(updates[cand].norm()),
                                                 "action_change_vs_control": act_change,
                                                 "mv_signal": mv, "hist_signal": hist})
                                env.sim.set_state(state); env.sim.forward()
                        # advance one replan with frozen policy
                        raw = policy_observation(obs, task.language, policy)
                        prep = prepare_observation(raw, config=policy.config, normalizer=policy.normalizer,
                                                   device=torch.device(DEV))
                        rep = policy.representation_extractor(**prep)
                        with fork():
                            torch.manual_seed(seed + 7919 * branch)
                            with torch.no_grad():
                                sg.adapter.beta.zero_()
                                c = policy.action_generator.sample(rep)
                        prev_chunk = c.detach()
                        ctrl = policy.normalizer.unnormalize_action(c)[0].float().cpu().numpy()
                        for a in ctrl[:6]:
                            obs, reward, done, info2 = env.step(np.clip(a, -1.0, 1.0))
                            step += 1
                            if is_success(env, info2, reward, done) or done: break
                        branch += 1
                    env.close()
                    print("episode", domain, suite, tid, ep, flush=True)
    finally:
        sg.close()
    (args.out / "b2.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print("rows", len(rows), "seconds", time.time() - t0)


if __name__ == "__main__":
    main()
