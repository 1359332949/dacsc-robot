"""E1: Frozen CoRP degradation screen on a clean base task vs shift variants.

No training.  Loads the released checkpoint once, runs paired episodes with the
same initial state and action-noise seed, and reports success rates.
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import (DUMMY_ACTION, MAX_STEPS, SETTLING_STEPS, configure_libero,
                             is_success, policy_observation)
from corp import load_policy


def _seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _bddl_path(suite, index, task, virtual_view: bool = True):
    p = Path(suite.get_task_bddl_file_path(index))
    if virtual_view and "_view_" in task.name:
        # OffScreenRenderEnv parses the _view_ suffix and loads the base BDDL.
        return Path(p).parent / f"{task.name}.bddl"
    return p


def _run_episode(env, policy, task, states, ep, language, max_steps, seed):
    _seed(seed); env.seed(seed); env.reset()
    obs = env.set_init_state(states[ep])
    success = done = False
    steps = 0
    for _ in range(SETTLING_STEPS):
        obs, reward, done, info = env.step(DUMMY_ACTION)
        success = is_success(env, info, reward, done)
        if success or done:
            break
    while not success and not done and steps < max_steps:
        batch = policy_observation(obs, language, policy)
        chunk = policy.predict(batch)[0].float().cpu().numpy()
        for action in chunk:
            obs, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
            steps += 1
            success = is_success(env, info, reward, done)
            if success or done or steps >= max_steps:
                break
    return {"success": bool(success), "policy_steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "repos/CoRP/checkpoints/corp_libero.pt")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--clean-bddl", type=Path, required=True)
    ap.add_argument("--clean-init-file", type=Path, required=True)
    ap.add_argument("--clean-language", required=True)
    ap.add_argument("--shift-task", type=int, action="append", required=True,
                    help="repeat for each variant; label from task map/classification")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    class C: pass
    c = C(); c.libero_src = args.libero_src; c.libero_config_path = args.libero_config_path
    c.render_gpu = args.render_gpu; c.device = args.device
    configure_libero(c)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    clean_states = torch.load(args.clean_init_file, weights_only=False)
    if args.episodes > len(clean_states):
        raise SystemExit(f"only {len(clean_states)} clean init states")
    policy = load_policy(args.checkpoint, device=args.device); policy.eval()
    max_steps = MAX_STEPS[args.suite]
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    summary = {"suite": args.suite, "clean_language": args.clean_language,
               "episodes_per_condition": args.episodes, "seed": args.seed,
               "conditions": {}}

    clean_env = OffScreenRenderEnv(bddl_file_name=str(args.clean_bddl),
                                   camera_heights=256, camera_widths=256,
                                   render_gpu_device_id=args.render_gpu)
    try:
        clean_rows = []
        for ep in range(args.episodes):
            row = _run_episode(clean_env, policy, suite.get_task(0), clean_states, ep,
                               args.clean_language, max_steps, args.seed + ep)
            row.update(condition="clean", episode=ep)
            clean_rows.append(row); records.append(row)
            print(f"clean ep={ep} success={int(row['success'])} steps={row['policy_steps']}", flush=True)
        clean_sr = sum(r["success"] for r in clean_rows) / len(clean_rows)
        summary["conditions"]["clean"] = {"success_rate": clean_sr, "episodes": clean_rows}

        for tid in args.shift_task:
            shift_task = suite.get_task(tid)
            shift_bddl = _bddl_path(suite, tid, shift_task)
            env = OffScreenRenderEnv(bddl_file_name=str(shift_bddl),
                                     camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
            try:
                rows = []
                for ep in range(args.episodes):
                    row = _run_episode(env, policy, shift_task, clean_states, ep,
                                       args.clean_language, max_steps, args.seed + ep)
                    row.update(condition=f"shift_{tid}", episode=ep, shift_task=tid,
                               shift_name=shift_task.name)
                    rows.append(row); records.append(row)
                    print(f"shift {tid} ep={ep} success={int(row['success'])} steps={row['policy_steps']}", flush=True)
                sr = sum(r["success"] for r in rows) / len(rows)
                summary["conditions"][f"shift_{tid}"] = {
                    "success_rate": sr, "delta_vs_clean": sr - clean_sr,
                    "shift_name": shift_task.name, "episodes": rows}
            finally:
                env.close()
    finally:
        clean_env.close()
    with (args.out / "episodes.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v["success_rate"] for k, v in summary["conditions"].items()}, indent=2))


if __name__ == "__main__":
    main()
