"""B1 capture v2 (24 episodes): per-replan images/actions/poses/geometry + image manifest."""

from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero, policy_observation, is_success  # noqa: E402
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.sg_cpa import SGCpa  # noqa: E402
from sg_cpa_phase4_final import CKPT, A_FILE  # noqa: E402

DEV = "cuda:0"
# fixed task selection: first three tasks per (domain, suite) cell from the frozen phase-3 config
CELLS = {
    ("Camera", "libero_spatial"): [705, 620, 671],
    ("Background", "libero_spatial"): [11, 19, 18],
    ("Camera", "libero_object"): [653, 701, 901],
    ("Background", "libero_object"): [3, 8, 18],
}


def fork():
    return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])


def save_jpg(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path, quality=92)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": h, "bytes": path.stat().st_size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/validation/b1_capture_v2"))
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--exec-len", type=int, default=6)
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
    rows, manifest = [], []
    t0 = time.time()
    try:
        for (domain, suite), tids in CELLS.items():
            suite_obj = suites[suite]
            for tid in tids:
                task = suite_obj.get_task(tid)
                bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                    weights_only=False)
                base_name = task.bddl_file.split("_view_")[0] + ".bddl"
                base_path = Path(get_libero_path("bddl_files")) / task.problem_folder / base_name
                goal_text = base_path.read_text() if base_path.exists() else ""
                for ep in range(args.episodes):
                    seed = 30 + tid * 1000 + ep
                    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                             render_gpu_device_id=0)
                    np.random.seed(seed); torch.manual_seed(seed); env.seed(seed); env.reset()
                    obs = env.set_init_state(states[ep % len(states)])
                    for _ in range(SETTLING_STEPS):
                        obs, _, done, _ = env.step(DUMMY_ACTION)
                        if done: break
                    run_dir = args.out / domain / suite / f"task{tid}_ep{ep}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    body_names = [str(x) for x in env.sim.model.body_names]
                    geom_names = [str(x) for x in env.sim.model.geom_names]
                    prev_chunk = None
                    steps = done = success = False
                    replan = 0; steps = 0
                    while not success and not done and steps < args.max_steps:
                        raw = policy_observation(obs, task.language, policy)
                        prep = prepare_observation(raw, config=policy.config, normalizer=policy.normalizer,
                                                   device=torch.device(DEV))
                        rep = policy.representation_extractor(**prep)
                        action_seed = seed + 7919 * replan
                        with fork():
                            torch.manual_seed(action_seed)
                            chunk = policy.action_generator.sample(rep)
                        views = sg.make_view_images(prep, action_seed + 1)
                        view_actions = []
                        for img in views:
                            with fork():
                                torch.manual_seed(action_seed)
                                with torch.no_grad():
                                    r = policy.representation_extractor(img, prep["state"], prep["instruction"])
                                    view_actions.append(policy.action_generator.sample(r))
                        pair = [float((view_actions[i] - view_actions[j]).abs().mean())
                                for i in range(4) for j in range(i + 1, 4)]
                        ctrl = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
                        img_entries = {}
                        for vname in raw["images"]:
                            arr = np.asarray(raw["images"][vname])
                            if arr.ndim == 4: arr = arr[0]
                            if arr.ndim == 3 and arr.shape[0] == 3: arr = arr.transpose(1, 2, 0)
                            p = run_dir / f"replan{replan:03d}_{vname}.jpg"
                            img_entries[vname] = save_jpg(arr, p)
                        body_pos = {n: [float(x) for x in env.sim.data.body_xpos[i]] for i, n in enumerate(body_names)}
                        body_quat = {n: [float(x) for x in env.sim.data.body_xquat[i]] for i, n in enumerate(body_names)}
                        geom_pos = [[float(x) for x in g] for g in env.sim.data.geom_xpos]
                        geom_size = [[float(x) for x in g] for g in env.sim.model.geom_size]
                        geom_bodyid = [int(x) for x in env.sim.model.geom_bodyid]
                        hist = None
                        if prev_chunk is not None:
                            hist = float((prev_chunk[0, args.exec_len:] - chunk[0, :args.exec_len]).abs().mean())
                        rec = {"domain": domain, "suite": suite, "task_id": tid, "episode": ep,
                               "seed": seed, "action_seed": action_seed, "replan": replan, "steps": steps,
                               "state": [float(x) for x in np.asarray(raw["state"]).reshape(-1)],
                               "instruction": task.language, "beta": 0.0,
                               "action_chunk_norm": chunk[0].float().cpu().tolist(),
                               "action_chunk_ctrl": ctrl.tolist(),
                               "view_actions": [v[0].float().cpu().tolist() for v in view_actions],
                               "view_pairwise_distances": pair,
                               "history_leftover_distance": hist,
                               "images": img_entries,
                               "body_names": body_names, "body_pos": body_pos, "body_quat": body_quat,
                               "geom_names": geom_names, "geom_pos": geom_pos,
                               "geom_size": geom_size, "geom_bodyid": geom_bodyid,
                               "goal_text": goal_text, "success_so_far": False}
                        rows.append(rec)
                        manifest.extend(img_entries.values())
                        prev_chunk = chunk.detach()
                        for a in ctrl[:args.exec_len]:
                            obs, reward, done, info = env.step(np.clip(a, -1.0, 1.0))
                            steps += 1
                            success = is_success(env, info, reward, done)
                            if success or done or steps >= args.max_steps: break
                        rec["success_so_far"] = bool(success)
                        if not (success or done) and steps < args.max_steps:
                            raw2 = policy_observation(obs, task.language, policy)
                            for vname in raw2["images"]:
                                arr = np.asarray(raw2["images"][vname])
                                if arr.ndim == 4: arr = arr[0]
                                if arr.ndim == 3 and arr.shape[0] == 3: arr = arr.transpose(1, 2, 0)
                                p = run_dir / f"replan{replan:03d}_{vname}_next.jpg"
                                ent = save_jpg(arr, p)
                                rec.setdefault("next_images", {})[vname] = ent
                                manifest.append(ent)
                        replan += 1
                    rows.append({"_episode_summary": {"domain": domain, "suite": suite, "task_id": tid,
                                                      "episode": ep, "seed": seed,
                                                      "success": bool(success), "steps": steps,
                                                      "replans": replan}})
                    env.close()
                    print("episode", domain, suite, tid, ep, "success", success, "steps", steps, flush=True)
    finally:
        sg.close()
    (args.out / "capture.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (args.out / "image_manifest.json").write_text(json.dumps({"root": str(args.out), "images": manifest}, indent=2))
    print("rows", len(rows), "images", len(manifest), "seconds", time.time() - t0)


if __name__ == "__main__":
    main()
