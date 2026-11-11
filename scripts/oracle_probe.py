"""Oracle identifiability probe (reviewer experiment A).

For each camera-perturbed task we collect PAIRED writes at identical simulator
states:
  1. rollout in the perturbed (camera) env, record executed actions and the
     last-Euler write mean at every replan;
  2. replay the SAME actions from the SAME init state in the clean env (same
     task, original bddl): camera perturbations do not change physics, so the
     states are identical -> w_clean[i] is a counterfactual for w_shift[i];
  3. Oracle shift:  dw[i] = w_shift[i] - w_clean[i];
  4. evaluate the perturbed env with the oracle correction  w' = w - alpha*dw[i]
     and compare success with the uncorrected run of step 1.

If the oracle correction improves success -> the estimator is the bottleneck.
If not -> the interface site / correction hypothesis is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import (  # noqa: E402
    DUMMY_ACTION, MAX_STEPS, SETTLING_STEPS, configure_libero, is_success, policy_observation,
)
from corp import load_policy  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402

CLEAN_ROOT = Path("/hy-tmp/dacsc-robot-784fe2e/repos/CoRP/third_party/LIBERO")


class Capture:
    def __init__(self, attention, num_steps: int, horizon: int):
        self.attention = attention
        self.original = attention.attend
        self.num_steps = num_steps
        self.horizon = horizon
        self.step = 0
        self.predicts = 0
        self.writes: list[np.ndarray] = []
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        if out.shape[1] != self.horizon:
            self.step = 0
            self.predicts += 1
            return out
        self.step += 1
        if self.step == self.num_steps:
            self.writes.append(out.detach().reshape(-1, out.shape[-1]).mean(dim=0).float().cpu().numpy())
        return out

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


class Oracle:
    def __init__(self, attention, num_steps: int, horizon: int, shifts: list[np.ndarray], alpha: float):
        self.attention = attention
        self.original = attention.attend
        self.num_steps = num_steps
        self.horizon = horizon
        self.shifts = shifts
        self.alpha = float(alpha)
        self.step = 0
        self.predicts = 0
        attention.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        if out.shape[1] != self.horizon:
            self.step = 0
            self.predicts += 1
            return out
        self.step += 1
        if self.step == self.num_steps and self.predicts - 1 < len(self.shifts):
            dw = self.shifts[self.predicts - 1]
            corr = torch.tensor(-self.alpha * dw, dtype=out.dtype, device=out.device)
            return out + corr
        return out

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]

    def reset(self):
        self.step = 0
        self.predicts = 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--save-pairs", type=Path, default=None,
                    help="npz with per-task w_shift / w_clean arrays for capacity analysis")
    ap.add_argument("--capture-only", action="store_true",
                    help="only collect paired writes (skip the oracle-corrected rollout)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-episodes", type=int, default=1)
    ap.add_argument("--clean-root", type=Path, default=None)
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    verify_release_checkpoint(_CORP_REPO / "checkpoints/corp_libero.pt")
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy = load_policy(_CORP_REPO / "checkpoints/corp_libero.pt", device="cuda:0")
    num_steps = int(policy.config["action_generator"]["num_sample_steps"])
    horizon = int(policy.config["action_horizon"])
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task_ids = []
    for part in args.tasks.split(","):
        if "-" in part:
            lo, hi = part.split("-"); task_ids.extend(range(int(lo), int(hi) + 1))
        else:
            task_ids.append(int(part))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        plus_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        initial_states = torch.load(states_path, weights_only=False)
        name = task.name.split("_view_")[0] if "_view_" in task.name else task.name
        clean_root = Path(args.clean_root) if getattr(args, "clean_root", None) else CLEAN_ROOT
        clean_bddl = clean_root / "libero" / "libero" / "bddl_files" / task.problem_folder / f"{name}.bddl"
        if not clean_bddl.exists():
            print(f"task={task_id} skip (no clean bddl {clean_bddl})", flush=True); continue
        seed = args.seed + task_id * 1000
        # ---- 1) perturbed rollout + capture
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        env = OffScreenRenderEnv(bddl_file_name=str(plus_bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
        env.seed(seed); env.reset(); obs = env.set_init_state(initial_states[0])
        for _ in range(SETTLING_STEPS):
            obs, reward, done, info = env.step(DUMMY_ACTION)
            if is_success(env, info, reward, done) or done: break
        cap = Capture(policy.action_generator.blocks[0].attention, num_steps, horizon)
        executed, base_success, steps = [], False, 0
        while not base_success and not done and steps < MAX_STEPS["libero_spatial"]:
            batch = policy_observation(obs, task.language, policy)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    chunk = policy.predict(batch)[0].float().cpu().numpy()
            for a in chunk:
                executed.append(a)
                obs, reward, done, info = env.step(np.clip(a, -1.0, 1.0))
                steps += 1
                base_success = is_success(env, info, reward, done)
                if base_success or done or steps >= MAX_STEPS["libero_spatial"]: break
        w_shift = [w.copy() for w in cap.writes]
        cap.close(); env.close()
        # ---- 2) clean replay of the same actions from the same init state
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        cenv = OffScreenRenderEnv(bddl_file_name=str(clean_bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
        cenv.seed(seed); cenv.reset(); cobs = cenv.set_init_state(initial_states[0])
        for _ in range(SETTLING_STEPS):
            cobs, reward, done, info = cenv.step(DUMMY_ACTION)
            if done: break
        ccap = Capture(policy.action_generator.blocks[0].attention, num_steps, horizon)
        for i in range(0, len(executed), horizon):
            batch = policy_observation(cobs, task.language, policy)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    policy.predict(batch)
            for a in executed[i:i + horizon]:
                cobs, reward, done, info = cenv.step(np.clip(a, -1.0, 1.0))
                if done: break
            if done: break
        w_clean = [w.copy() for w in ccap.writes]
        ccap.close(); cenv.close()
        n = min(len(w_shift), len(w_clean))
        shifts = [w_shift[i] - w_clean[i] for i in range(n)]
        if args.save_pairs is not None:
            store = {}
            if args.save_pairs.exists():
                store = dict(np.load(args.save_pairs))
            store[f"shift_{task_id}"] = np.stack([w_shift[i] for i in range(n)]).astype(np.float32)
            store[f"clean_{task_id}"] = np.stack([w_clean[i] for i in range(n)]).astype(np.float32)
            np.savez(args.save_pairs, **store)
        if args.capture_only:
            print({"task_id": task_id, "pairs": n, "capture_only": True}, flush=True)
            continue
        # ---- 3) oracle-corrected evaluation
        np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        env = OffScreenRenderEnv(bddl_file_name=str(plus_bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
        env.seed(seed); env.reset(); obs = env.set_init_state(initial_states[0])
        for _ in range(SETTLING_STEPS):
            obs, reward, done, info = env.step(DUMMY_ACTION)
            if is_success(env, info, reward, done) or done: break
        oracle = Oracle(policy.action_generator.blocks[0].attention, num_steps, horizon, shifts, args.alpha)
        corr_success, steps = False, 0
        while not corr_success and not done and steps < MAX_STEPS["libero_spatial"]:
            batch = policy_observation(obs, task.language, policy)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    chunk = policy.predict(batch)[0].float().cpu().numpy()
            for a in chunk:
                obs, reward, done, info = env.step(np.clip(a, -1.0, 1.0))
                steps += 1
                corr_success = is_success(env, info, reward, done)
                if corr_success or done or steps >= MAX_STEPS["libero_spatial"]: break
        oracle.close(); env.close()
        rec = {"task_id": task_id, "pairs": n, "base_success": bool(base_success),
               "oracle_success": bool(corr_success), "alpha": args.alpha}
        records.append(rec)
        with args.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
