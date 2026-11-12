"""B1 data capture: frozen CoRP observation/action stream for prospective signal comparison.

Per replan records: o_t (both cameras, JPEG), state, instruction, action chunk (frozen,
fixed latent), multi-view actions, executed actions, o_{t+1}, body positions/names,
base bddl goal text, and end-of-episode success.  No adaptation, no success-rate evaluation.
"""

from __future__ import annotations
import argparse, json, sys, time
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
CELLS = {
    ("Camera", "libero_spatial"): [705, 620],
    ("Background", "libero_spatial"): [11, 19],
    ("Camera", "libero_object"): [653, 701],
    ("Background", "libero_object"): [3, 8],
}


def fork():
    return torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else [])


def sample(policy, rep, seed):
    with fork():
        torch.manual_seed(int(seed))
        with torch.no_grad():
            return policy.action_generator.sample(rep).detach()


def slice_view(batch, name):
    return {"images": {name: batch["images"][name]}, "state": batch["state"],
            "instruction": batch["instruction"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/validation/b1_capture"))
    ap.add_argument("--episodes", type=int, default=1)
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
    sg = SGCpa(policy, a_basis, seed=1234)  # only used for view transforms/target sampling
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in ("libero_spatial", "libero_object")}
    rows = []
    t0 = time.time()
    try:
        for (domain, suite), tids in CELLS.items():
            suite_obj = suites[suite]
            for tid in tids:
                task = suite_obj.get_task(tid)
                bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                    weights_only=False)
                # base bddl goal text (strip _view_... suffix)
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
                    prev_chunk = None
                    steps = 0; done = False; success = False; replan = 0
                    while not success and not done and steps < args.max_steps:
                        raw = policy_observation(obs, task.language, policy)
                        prep = prepare_observation(raw, config=policy.config, normalizer=policy.normalizer,
                                                   device=torch.device(DEV))
                        rep = policy.representation_extractor(**prep)
                        action_seed = seed + 7919 * replan
                        with fork():
                            torch.manual_seed(action_seed)
                            chunk = policy.action_generator.sample(rep)  # [1,12,7] normalized
                        views = sg.make_view_images(prep, action_seed + 1)
                        view_actions = []
                        for img in views:
                            with fork():
                                torch.manual_seed(action_seed)
                                with torch.no_grad():
                                    r = policy.representation_extractor(img, prep["state"], prep["instruction"])
                                    view_actions.append(policy.action_generator.sample(r))
                        dview = float(np.mean([float((view_actions[i] - view_actions[j]).abs().mean())
                                               for i in range(4) for j in range(i + 1, 4)]))
                        ctrl = policy.normalizer.unnormalize_action(chunk)[0].float().cpu().numpy()
                        # save images (o_t) as JPEG
                        img_paths = {}
                        for vname in raw["images"]:
                            arr = np.asarray(raw["images"][vname])
                            if arr.ndim == 4:
                                arr = arr[0]
                            if arr.ndim == 3 and arr.shape[0] == 3:
                                arr = arr.transpose(1, 2, 0)
                            arr = arr.astype(np.uint8)
                            p = run_dir / f"replan{replan:03d}_{vname}.jpg"
                            Image.fromarray(arr).save(p, quality=92)
                            img_paths[vname] = str(p)
                        body_pos = {n: [float(x) for x in env.sim.data.body_xpos[i]]
                                    for i, n in enumerate(body_names)}
                        hist = None
                        if prev_chunk is not None:
                            hist = float((prev_chunk[0, args.exec_len:] - chunk[0, :args.exec_len]).abs().mean())
                        rec = {"domain": domain, "suite": suite, "task_id": tid, "episode": ep,
                               "seed": seed, "action_seed": action_seed,
                               "replan": replan, "steps": steps,
                               "state": [float(x) for x in np.asarray(raw["state"]).reshape(-1)],
                               "instruction": task.language, "beta": 0.0,
                               "action_chunk_norm": chunk[0].float().cpu().tolist(),
                               "action_chunk_ctrl": ctrl.tolist(),
                               "view_actions_mean_distance": dview,
                               "history_leftover_distance": hist,
                               "images": img_paths,
                               "body_names": body_names, "body_pos": body_pos,
                               "goal_text": goal_text,
                               "success_so_far": False}
                        rows.append(rec)
                        # execute
                        prev_chunk = chunk.detach()
                        for a in ctrl[:args.exec_len]:
                            obs, reward, done, info = env.step(np.clip(a, -1.0, 1.0))
                            steps += 1
                            success = is_success(env, info, reward, done)
                            if success or done or steps >= args.max_steps: break
                        rec["success_so_far"] = bool(success)
                        # next observation images
                        if not (success or done) and steps < args.max_steps:
                            raw2 = policy_observation(obs, task.language, policy)
                            for vname in raw2["images"]:
                                arr = np.asarray(raw2["images"][vname])
                                if arr.ndim == 4:
                                    arr = arr[0]
                                if arr.ndim == 3 and arr.shape[0] == 3:
                                    arr = arr.transpose(1, 2, 0)
                                arr = arr.astype(np.uint8)
                                p = run_dir / f"replan{replan:03d}_{vname}_next.jpg"
                                Image.fromarray(arr).save(p, quality=92)
                                rec.setdefault("next_images", {})[vname] = str(p)
                        replan += 1
                    rec_ep = {"domain": domain, "suite": suite, "task_id": tid, "episode": ep,
                              "seed": seed, "success": bool(success), "steps": steps,
                              "replans": replan}
                    rows.append({"_episode_summary": rec_ep})
                    env.close()
                    print("episode", domain, suite, tid, ep, "success", success, "steps", steps, flush=True)
    finally:
        sg.close()
    (args.out / "capture.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print("rows", len(rows), "seconds", time.time() - t0)


if __name__ == "__main__":
    main()
