"""B2 protocol audits (preflight only; no formal counterfactual experiment).

Modes:
  targets : parse BDDL goal literals for the 12 tasks and verify the mapped object/target pair
  restore : environment state save/restore audit (obs, body poses, counters, RNG)
  zero24  : control vs zero-update over 4x6-step replans, trajectory equality + delta noise
  stage   : mechanical grasp-stage audit at replans 8/12 on the 24 baseline episodes
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import re

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
PAT = re.compile(r"\((On|In)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)")


def setup():
    class C: pass
    c = C(); c.libero_src = Path("/hy-tmp/libero_plus"); c.libero_config_path = Path("/hy-tmp/libero_plus_config")
    c.render_gpu = 0; c.device = DEV
    configure_libero(c)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    policy = load_policy(CKPT, device=DEV)
    a_basis = torch.tensor(json.loads(A_FILE.read_text())["action_subspace"], dtype=torch.float64)
    return policy, a_basis, benchmark, get_libero_path, OffScreenRenderEnv


def canonical_goal(task, folder):
    import re as _re
    for pat in (r"_(table|tb)_\d+$", r"_view_.*$", r"_add_\d+$", r"_(light|noise|background).*$"):
        base = _re.sub(pat, "", task.name)
        if base != task.name:
            p = folder / (base + ".bddl")
            if p.exists():
                return p.read_text()
    parts = task.name.split("_")
    for k in range(len(parts), 0, -1):
        p = folder / ("_".join(parts[:k]) + ".bddl")
        if p.exists():
            return p.read_text()
    return ""


def expected_literal(suite, instruction):
    if suite == "libero_spatial":
        return ("On", "akita_black_bowl_1", "plate_1")
    instr = instruction.lower()
    for obj in ("alphabet_soup_1", "salad_dressing_1", "butter_1"):
        if obj.replace("_", " ") in instr or obj.split("_")[0] in instr:
            return ("In", obj, "basket_1_contain_region")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/validation/b2_audit"))
    ap.add_argument("--branch", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    policy, a_basis, benchmark, get_libero_path, OffScreenRenderEnv = setup()
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in ("libero_spatial", "libero_object")}
    if args.mode == "targets":
        res = []
        for (domain, suite), tids in CELLS.items():
            for tid in tids:
                task = suites[suite].get_task(tid)
                folder = Path(get_libero_path("bddl_files")) / task.problem_folder
                g = canonical_goal(task, folder)
                lits = PAT.findall(g)
                exp = expected_literal(suite, task.language)
                task_lits = [x for x in lits if "floor_" not in x[2]]
                ok = exp in task_lits or (exp and exp[2].replace("_contain_region", "") and
                                          any(x[0] == exp[0] and x[1] == exp[1] and
                                              x[2].replace("_contain_region", "") == exp[2].replace("_contain_region", "")
                                              for x in task_lits))
                res.append({"domain": domain, "suite": suite, "task_id": tid,
                            "instruction": task.language, "expected": exp,
                            "task_literals": task_lits, "match": bool(ok)})
                print(tid, suite, "OK" if ok else "MISMATCH", exp, task_lits)
        (args.out / "targets.json").write_text(json.dumps(res, indent=2))
        print("mismatches", sum(1 for r in res if not r["match"]))
        return 0
    # env-based modes use one representative task per domain
    sel = [("Camera", "libero_spatial", 705), ("Background", "libero_object", 3)]
    if args.mode == "restore":
        res = []
        for domain, suite, tid in sel:
            task = suites[suite].get_task(tid)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
            np.random.seed(7); torch.manual_seed(7); env.seed(7); env.reset()
            obs = env.set_init_state(states[0])
            for _ in range(SETTLING_STEPS): obs, _, done, _ = env.step(DUMMY_ACTION)
            before = {n: np.array(env.sim.data.body_xpos[i]) for i, n in enumerate(env.sim.model.body_names)}
            st = env.sim.get_state()
            np_state = np.random.get_state(); th_state = torch.get_rng_state()
            timestep = getattr(env, "timestep", None)
            for _ in range(6): env.step(DUMMY_ACTION)
            after = {n: np.array(env.sim.data.body_xpos[i]) for i, n in enumerate(env.sim.model.body_names)}
            env.sim.set_state(st); env.sim.forward()
            restored = {n: np.array(env.sim.data.body_xpos[i]) for i, n in enumerate(env.sim.model.body_names)}
            max_diff = max(float(np.abs(before[n] - restored[n]).max()) for n in before)
            moved = max(float(np.abs(before[n] - after[n]).max()) for n in before)
            res.append({"domain": domain, "suite": suite, "task_id": tid,
                        "restore_max_body_diff": max_diff, "moved_before_restore": moved,
                        "timestep_saved": timestep, "timestep_after": getattr(env, "timestep", None),
                        "np_rng_equal": all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), th_state)) if False else None})
            env.close()
        (args.out / "restore.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return 0
    if args.mode == "determinism":
        res = []
        for domain, suite, tid in sel:
            task = suites[suite].get_task(tid)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                weights_only=False)
            envs = []
            for _ in range(2):
                e = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                       render_gpu_device_id=0)
                np.random.seed(30 + tid * 1000); torch.manual_seed(30 + tid * 1000); e.seed(30 + tid * 1000); e.reset()
                o = e.set_init_state(states[0])
                for _ in range(SETTLING_STEPS): o, _, d, _ = e.step(DUMMY_ACTION)
                envs.append(e)
            # same action sequence: 8 dummy actions
            seq = [DUMMY_ACTION.copy() for _ in range(8)]
            traj = [[], []]
            for k, a in enumerate(seq):
                for i, e in enumerate(envs):
                    e.step(a)
                    traj[i].append({n: np.array(e.sim.data.body_xpos[j]) for j, n in enumerate(e.sim.model.body_names)})
            md = max(float(np.abs(traj[0][k][n] - traj[1][k][n]).max())
                     for k in range(len(seq)) for n in traj[0][k])
            res.append({"domain": domain, "suite": suite, "task_id": tid, "max_body_diff": md})
            for e in envs: e.close()
        (args.out / "determinism.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return 0
    if args.mode == "zero24":
        res = []
        sg = SGCpa(policy, a_basis, seed=1234)
        try:
            for domain, suite, tid in sel:
                task = suites[suite].get_task(tid)
                bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                    weights_only=False)
                seed = 30 + tid * 1000
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                         render_gpu_device_id=0)
                np.random.seed(seed); torch.manual_seed(seed); env.seed(seed); env.reset()
                obs = env.set_init_state(states[0])
                for _ in range(SETTLING_STEPS): obs, _, done, _ = env.step(DUMMY_ACTION)
                names = [str(x) for x in env.sim.model.body_names]
                obj = "akita_black_bowl_1_main" if suite == "libero_spatial" else "alphabet_soup_1_main"
                oi = names.index(obj); ti = names.index("plate_1_main" if suite == "libero_spatial" else "basket_1_main")
                pair = (names[oi], names[ti])
                step = 0; branch = 0; done = False
                while not done and branch < args.branch and step < 240:
                    raw = policy_observation(obs, task.language, policy)
                    prep = prepare_observation(raw, config=policy.config, normalizer=policy.normalizer,
                                               device=torch.device(DEV))
                    rep = policy.representation_extractor(**prep)
                    with torch.no_grad():
                        sg.adapter.beta.zero_(); torch.manual_seed(seed + 7919 * branch)
                        c = policy.action_generator.sample(rep)
                    ctrl = policy.normalizer.unnormalize_action(c)[0].float().cpu().numpy()
                    for a in ctrl[:6]:
                        obs, reward, done, info = env.step(np.clip(a, -1.0, 1.0))
                        step += 1
                        if is_success(env, info, reward, done) or done: break
                    branch += 1
                d0 = float(np.linalg.norm(np.array(env.sim.data.body_xpos[oi]) - np.array(env.sim.data.body_xpos[ti])))
                state = env.sim.get_state()
                for mode in ("control", "zero"):
                    env.sim.set_state(state); env.sim.forward()
                    traj = []
                    for k in range(4):
                        raw = policy_observation(obs, task.language, policy)
                        prep = prepare_observation(raw, config=policy.config, normalizer=policy.normalizer,
                                                   device=torch.device(DEV))
                        rep = policy.representation_extractor(**prep)
                        with torch.no_grad():
                            sg.adapter.beta.zero_(); torch.manual_seed(seed + 7919 * (branch + k))
                            c = policy.action_generator.sample(rep)
                        ctrl = policy.normalizer.unnormalize_action(c)[0].float().cpu().numpy()
                        for a in ctrl[:6]:
                            obs, reward, done, info = env.step(np.clip(a, -1.0, 1.0))
                            traj.append([float(x) for x in env.sim.data.body_xpos[oi]])
                            if is_success(env, info, reward, done) or done: break
                        if done: break
                    pa = np.array(traj)
                    res.append({"domain": domain, "suite": suite, "task_id": tid, "mode": mode,
                                "d0": d0, "n_steps": len(traj),
                                "p_final": float(1 - np.linalg.norm(pa[-1] - np.array(env.sim.data.body_xpos[ti])) / d0),
                                "traj": traj})
                env.close()
        finally:
            sg.close()
        ctrl = [r for r in res if r["mode"] == "control"]; zero = [r for r in res if r["mode"] == "zero"]
        diffs = []
        for c, z in zip(ctrl, zero):
            L = min(len(c["traj"]), len(z["traj"]))
            diffs.append(max(float(np.abs(np.array(c["traj"][i]) - np.array(z["traj"][i])).max()) for i in range(L)))
        out = {"max_traj_diff": max(diffs) if diffs else None,
               "max_dp_diff": max(abs(c["p_final"] - z["p_final"]) for c, z in zip(ctrl, zero)) if ctrl else None,
               "records": res}
        (args.out / "zero24.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({k: out[k] for k in ("max_traj_diff", "max_dp_diff")}, indent=2))
        return 0
    if args.mode == "stage":
        res = []
        sg = SGCpa(policy, a_basis, seed=1234)
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
                        for _ in range(SETTLING_STEPS): obs, _, done, _ = env.step(DUMMY_ACTION)
                        names = [str(x) for x in env.sim.model.body_names]
                        # target object body
                        obj = ("akita_black_bowl_1_main" if suite == "libero_spatial" else None)
                        if obj is None:
                            for cand in ("alphabet_soup_1_main", "salad_dressing_1_main", "butter_1_main"):
                                if task.language.lower().replace("_", " ").find(cand.split("_")[0]) >= 0:
                                    obj = cand; break
                        oi = names.index(obj); gi = names.index("gripper0_eef")
                        obj0 = np.array(env.sim.data.body_xpos[oi])
                        step = 0; branch = 0; done = False
                        while not done and branch <= max(8, 12) and step < 240:
                            if branch in (8, 12):
                                eef = np.array(env.sim.data.body_xpos[gi])
                                op = np.array(env.sim.data.body_xpos[oi])
                                res.append({"domain": domain, "suite": suite, "task_id": tid, "episode": ep,
                                            "branch": branch, "step": step,
                                            "eef_obj_dist": float(np.linalg.norm(eef - op)),
                                            "obj_lift": float(op[2] - obj0[2])})
                            raw = policy_observation(obs, task.language, policy)
                            prep = prepare_observation(raw, config=policy.config, normalizer=policy.normalizer,
                                                       device=torch.device(DEV))
                            rep = policy.representation_extractor(**prep)
                            with torch.no_grad():
                                torch.manual_seed(seed + 7919 * branch)
                                sg.adapter.beta.zero_()
                                c = policy.action_generator.sample(rep)
                            ctrl = policy.normalizer.unnormalize_action(c)[0].float().cpu().numpy()
                            for a in ctrl[:6]:
                                obs, reward, done, info = env.step(np.clip(a, -1.0, 1.0))
                                step += 1
                                if is_success(env, info, reward, done) or done: break
                            branch += 1
                        env.close()
                        print("stage", domain, suite, tid, ep, "branches", branch)
        finally:
            sg.close()
        (args.out / "stage.json").write_text(json.dumps(res, indent=2))
        print("records", len(res))
        return 0
    raise SystemExit("unknown mode")


if __name__ == "__main__":
    raise SystemExit(main())
