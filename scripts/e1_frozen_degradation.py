"""E1: Frozen CoRP heterogeneous-domain degradation screen.

Per base task: Clean episodes + 4 domains x 2 variants.  Screening only; no
adaptation, no Gate, no repaired-policy rollout.
"""

from __future__ import annotations
import argparse, hashlib, json, re, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from corp import load_policy  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402
from evaluate_libero import (DUMMY_ACTION, MAX_STEPS, SETTLING_STEPS,  # noqa: E402
                             is_success, policy_observation)
from dacsc.libero_noise import apply_agentview_noise  # noqa: E402


def configure(args):
    import os, yaml
    src = args.libero_src.expanduser().resolve(); root = src / "libero" / "libero"
    sys.path.insert(0, str(src))
    cfg = args.libero_config.expanduser().resolve(); cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "datasets").mkdir(exist_ok=True)
    (cfg / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"), "assets": str(root / "assets"),
        "datasets": str(cfg / "datasets")}))
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def run_episode(policy, env, init_state, language, episode_seed, max_steps, precision, sensor_noise=0,
                reset_env=True):
    np.random.seed(episode_seed); torch.manual_seed(episode_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(episode_seed)
    env.seed(episode_seed)
    if reset_env:
        env.reset()
    observation = env.set_init_state(init_state)
    if sensor_noise:
        observation, _ = apply_agentview_noise(observation, sensor_noise, seed=episode_seed)
    success, done, steps = False, False, 0
    for _ in range(SETTLING_STEPS):
        observation, reward, done, info = env.step(DUMMY_ACTION)
        success = is_success(env, info, reward, done)
        if success or done: break
    while not success and not done and steps < max_steps:
        batch = policy_observation(observation, language, policy)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=precision == "bf16" and torch.cuda.is_available()):
            chunk = policy.predict(batch)[0].float().cpu().numpy()
        for action in chunk:
            observation, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
            steps += 1
            success = is_success(env, info, reward, done)
            if success or done or steps >= max_steps: break
    return {"success": bool(success), "steps": steps, "done": bool(done)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--task-limit", type=int, default=0)
    ap.add_argument("--task-index", type=int, default=-1)
    ap.add_argument("--domains", nargs="+", default=None)
    ap.add_argument("--cell-limit", type=int, default=0)
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--shift-only", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    tasks = matrix["base_tasks"][:args.task_limit or None]
    if args.task_index >= 0:
        tasks = [matrix["base_tasks"][args.task_index]]
    torch.set_num_threads(1)
    policy = load_policy(CKPT, device=args.device); policy.eval()
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to reuse non-empty output dir: {args.out} (pass --resume)")
    args.out.mkdir(parents=True, exist_ok=True)
    records = []; started = time.time()
    for bi, b in enumerate(tasks):
        init_states = torch.load(b["state_split"]["source_file"], weights_only=False)
        idxs = b["state_split"]["dev_indices"][:args.episodes]
        bddl_root = Path(b["variants"]["background"][0]["base_bddl"]).parent
        base_bddl = bddl_root / f"{b['base_task']}.bddl"
        base_language = BDDLUtils.get_problem_info(str(base_bddl))["language_instruction"]
        clean_cells = [("clean", None, None, None)]
        shift_cells = [(domain, v, domain, v["env_bddl_name"])
                       for domain, vs in b["variants"].items() for v in vs]
        if args.domains:
            shift_cells = [c for c in shift_cells if c[0] in set(args.domains)]
        if args.shift_only:
            cells = shift_cells
        elif args.clean_only:
            cells = clean_cells
        else:
            cells = clean_cells + shift_cells
        if args.num_shards > 1:
            cells = [c for j, c in enumerate(cells)
                     if (j + bi * 100) % args.num_shards == args.shard_index]
        if args.cell_limit:
            cells = cells[:args.cell_limit]
        for domain, v, _, env_bddl in cells:
            sensor_noise = 0
            if domain == "clean":
                env_bddl = str(base_bddl)
            elif domain == "sensor":
                m = re.search(r"_noise_(\d+)", v["name"]); sensor_noise = int(m.group(1)) if m else 0
            env = None
            if domain == "clean":
                env = OffScreenRenderEnv(bddl_file_name=env_bddl, camera_heights=256,
                                         camera_widths=256, render_gpu_device_id=0)
            try:
                for ep, idx in enumerate(idxs):
                    episode_seed = args.seed + bi * 1000 + ep
                    if domain == "clean":
                        run_env = env; reset_env = True
                    else:
                        # Shift variants must not be reset after construction:
                        # reset() reloads the base scene and can discard
                        # Background/Lighting/Camera assets.
                        run_env = OffScreenRenderEnv(bddl_file_name=env_bddl, camera_heights=256,
                                                     camera_widths=256, render_gpu_device_id=0)
                        reset_env = False
                    try:
                        r = run_episode(policy, run_env, init_states[idx], base_language,
                                        episode_seed, MAX_STEPS[b["suite"]], args.precision,
                                        sensor_noise=sensor_noise, reset_env=reset_env)
                    finally:
                        if domain != "clean":
                            run_env.close()
                    rec = {"suite": b["suite"], "base_task": b["base_task"], "domain": domain,
                           "variant_id": v["classification_id"] if v else None,
                           "benchmark_index": v.get("benchmark_index") if v else None,
                           "variant_name": v["name"] if v else None,
                           "init_index": int(idx), "episode_seed": episode_seed,
                           "precision": args.precision, **r}
                    records.append(rec)
                    with (args.out / "e1_frozen.jsonl").open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{b['suite']} {domain} {v['classification_id'] if v else 'clean'} "
                          f"ep={ep} success={int(r['success'])} steps={r['steps']}", flush=True)
            finally:
                if env is not None:
                    env.close()
    summary = {}
    for r in records:
        key = f"{r['suite']}::{r['base_task']}::{r['domain']}::{r['variant_id']}"
        s = summary.setdefault(key, {"n": 0, "success": 0, "steps": []})
        s["n"] += 1; s["success"] += int(r["success"]); s["steps"].append(r["steps"])
    for s in summary.values():
        s["success_rate"] = s["success"] / s["n"]; s["mean_steps"] = float(np.mean(s["steps"]))
    raw = args.out / "e1_frozen.jsonl"
    (args.out / "e1_summary.json").write_text(json.dumps({
        "episodes": args.episodes, "seed": args.seed, "precision": args.precision,
        "matrix_sha256": hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(Path(CKPT).read_bytes()).hexdigest(),
        "command": sys.argv, "raw_jsonl_sha256": hashlib.sha256(raw.read_bytes()).hexdigest() if raw.is_file() else None,
        "git_rev": __import__("subprocess").check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],stderr=__import__("subprocess").DEVNULL,text=True).strip(),
        "elapsed_seconds": time.time() - started, "cells": summary}, indent=2))
    print(json.dumps({"records": len(records), "cells": len(summary),
                      "elapsed": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
