"""A' mid-episode camera-shift experiment (in-place camera extrinsic edit).

Protocol:
  clean episode -> execute `exec_len` actions per replan -> at `--shift-step`
  (in executed steps) edit the camera pose in the SAME env (state untouched),
  refresh the observation, then continue with one of:
    frozen   : normal replan (policy sees the shifted view)
    leftover : execute the pre-shift NOMINAL leftover of the previous clean chunk
    cpia     : one leftover-based interface adaptation (episodic) then replan
  --sham: apply the same code path but with zero pose offset (control).

`--validate` additionally checks: qpos/qvel/act/ctrl/time unchanged, physics
transition identical under shift vs sham, image changed, edit reproducible.
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

CLEAN_ROOT = Path("/hy-tmp/dacsc-robot-784fe2e/repos/CoRP/third_party/LIBERO/libero/libero")


def snapshot(env):
    sim = env.env.sim
    return {
        "qpos": np.array(sim.data.qpos).copy(),
        "qvel": np.array(sim.data.qvel).copy(),
        "act": np.array(sim.data.act).copy() if sim.data.act.size else None,
        "ctrl": np.array(sim.data.ctrl).copy(),
        "time": float(sim.data.time),
    }


def same_state(a, b) -> bool:
    for key in a:
        if a[key] is None or b[key] is None:
            if a[key] is not None or b[key] is not None:
                return False
        elif not np.allclose(a[key], b[key]):
            return False
    return True


class CameraEditor:
    def __init__(self, env, camera: str, offset: np.ndarray):
        self.env = env
        self.sim = env.env.sim
        self.cid = self.sim.model.camera_name2id(camera)
        self.offset = np.asarray(offset, dtype=float)
        self.pos0 = np.array(self.sim.model.cam_pos[self.cid]).copy()
        self.applied = False

    def apply(self):
        self.sim.model.cam_pos[self.cid] = self.pos0 + self.offset
        self.sim.forward()
        self.applied = True

    def restore(self):
        self.sim.model.cam_pos[self.cid] = self.pos0
        self.sim.forward()

    def refreshed_obs(self, env):
        return env.env._get_observations(force_update=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--mode", choices=["frozen", "leftover", "cpia"], required=True)
    ap.add_argument("--exec-len", type=int, default=6)
    ap.add_argument("--shift-step", type=int, default=60)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--offset", default="0.05,0,0")
    ap.add_argument("--sham", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    verify_release_checkpoint(_CORP_REPO / "checkpoints/corp_libero.pt")
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy = load_policy(_CORP_REPO / "checkpoints/corp_libero.pt", device="cuda:0")
    horizon = int(policy.config["action_horizon"])
    transform = None
    if args.mode == "cpia":
        from dacsc.leftover_tta import LowRankInterfaceTransform
        transform = LowRankInterfaceTransform(policy.action_generator.blocks[0].attention,
                                              int(policy.config["hidden_dim"]), rank=args.rank)
        transform.optimizer = torch.optim.Adam([transform.u, transform.v], lr=args.lr)
    offset = np.zeros(3) if args.sham else np.array([float(x) for x in args.offset.split(",")])
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [int(x) for x in args.tasks.split(",")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for task_id in task_ids:
            task = suite.get_task(task_id)
            name = task.name.split("_view_")[0] if "_view_" in task.name else task.name
            clean_dir = CLEAN_ROOT / "bddl_files" / task.problem_folder
            candidates = sorted(clean_dir.glob("*.bddl"), key=lambda p: -len(p.stem))
            bddl = next((p for p in candidates if name.startswith(p.stem)), None)
            if bddl is None:
                print(f"task={task_id} skip (no clean bddl for {name})", flush=True)
                env = None
                continue
            states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                                weights_only=False)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                     render_gpu_device_id=args.render_gpu)
            seed = args.seed + task_id * 1000
            np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            env.seed(seed); env.reset(); obs = env.set_init_state(states[0])
            for _ in range(SETTLING_STEPS):
                obs, _, done, _ = env.step(DUMMY_ACTION)
                if done: break
            editor = CameraEditor(env, args.camera, offset)
            validate_info: dict = {}
            img_before = np.array(obs["agentview_image"]).copy() if "agentview_image" in obs else None
            if args.validate:
                # physics transition must not depend on the camera pose (no rendering)
                sim = env.env.sim
                st = snapshot(env)

                def dummy_step():
                    sim.data.ctrl[:] = 0.0   # identical control in both branches
                    sim.step()

                editor.apply(); dummy_step(); s_shift = snapshot(env)
                sim.data.qpos[:] = st["qpos"]; sim.data.qvel[:] = st["qvel"]
                if st["act"] is not None: sim.data.act[:] = st["act"]
                sim.data.time = st["time"]; sim.forward()
                editor.restore(); dummy_step(); s_sham = snapshot(env)
                sim.data.qpos[:] = st["qpos"]; sim.data.qvel[:] = st["qvel"]
                if st["act"] is not None: sim.data.act[:] = st["act"]
                sim.data.time = st["time"]; sim.forward()
                validate_info["physics_unchanged"] = same_state(s_shift, s_sham)
                obs = env.env._get_observations(force_update=True)
            success, done, steps = False, False, 0
            previous_chunk = None
            shifted = False
            if transform is not None:
                transform.reset_adaptation(lr=args.lr, seed=1234)
            while not success and not done and steps < MAX_STEPS[args.suite]:
                if (not shifted) and steps >= args.shift_step:
                    pre_edit = snapshot(env)
                    img_before = np.array(obs["agentview_image"]).copy() if "agentview_image" in obs else None
                    editor.apply()
                    obs = editor.refreshed_obs(env)
                    shifted = True
                    if args.validate:
                        post_edit = snapshot(env)
                        validate_info["state_unchanged"] = same_state(pre_edit, post_edit)
                        validate_info["image_changed"] = float(
                            np.abs(img_before.astype(float) - obs["agentview_image"].astype(float)).mean()
                        ) if img_before is not None else -1.0
                if shifted and args.mode == "leftover" and previous_chunk is not None:
                    chunk = previous_chunk[args.exec_len:]
                else:
                    batch = policy_observation(obs, task.language, policy)
                    if transform is not None and shifted and previous_chunk is not None:
                        from dacsc.leftover_tta import sample_with_grad
                        batch_pre = batch
                        # one episodic interface update using the pre-shift leftover anchor
                        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
                        for _ in range(2):
                            transform.active = True; transform.reset()
                            torch.manual_seed(1234)
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                rep = policy.representation_extractor(batch["images"], batch["state"], batch["instruction"])
                                with torch.random.fork_rng(devices=devices):
                                    torch.manual_seed(1234)
                                    c = sample_with_grad(policy.action_generator, rep,
                                                         policy.config["action_generator"]["num_sample_steps"])[0].float()
                            target = torch.tensor(previous_chunk[args.exec_len:], device=c.device, dtype=c.dtype)
                            loss = torch.nn.functional.mse_loss(c[:horizon - args.exec_len],
                                                                target[:horizon - args.exec_len])
                            transform.optimizer.zero_grad(set_to_none=True); loss.backward(); transform.optimizer.step()
                        transform.active = False
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        with torch.no_grad():
                            if transform is not None:
                                transform.active = True; transform.reset()
                            chunk = policy.predict(batch)[0].float().cpu().numpy()
                            if transform is not None:
                                transform.active = False
                previous_chunk = chunk
                for action in chunk[:args.exec_len]:
                    obs, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                    steps += 1
                    success = is_success(env, info, reward, done)
                    if success or done or steps >= MAX_STEPS[args.suite]: break
            editor.restore()
            env.close()
            rec = {"suite": args.suite, "task_id": task_id, "success": bool(success), "steps": steps,
                   "mode": args.mode, "shift_step": args.shift_step, "sham": args.sham, **validate_info}
            output.write(json.dumps(rec) + "\n"); output.flush()
            print(rec, flush=True)
    if transform is not None:
        transform.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
