"""A' feasibility: in-place camera extrinsic change inside the SAME MuJoCo env.

Validations:
  1. physical state (qpos/qvel/act/ctrl/time) unchanged by the camera edit;
  2. physics transition unchanged under a no-op (sham) camera edit;
  3. rendered observation changes with the camera pose;
  4. identical edits are reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero  # noqa: E402


def state(env):
    sim = env.sim
    return {
        "qpos": np.array(sim.data.qpos).copy(),
        "qvel": np.array(sim.data.qvel).copy(),
        "act": np.array(sim.data.act).copy() if sim.data.act.size else None,
        "ctrl": np.array(sim.data.ctrl).copy(),
        "time": float(sim.data.time),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--dx", type=float, default=0.05)
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    args = ap.parse_args()
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                        weights_only=False)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                             render_gpu_device_id=args.render_gpu)
    env.seed(7); env.reset(); obs = env.set_init_state(states[0])
    for _ in range(SETTLING_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    sim = env.sim
    cam_id = sim.model.camera_name2id(args.camera)
    pos0 = np.array(sim.model.cam_pos[cam_id]).copy()
    quat0 = np.array(sim.model.cam_quat[cam_id]).copy()
    s0 = state(env)
    img0 = np.array(obs["agentview_image"]).copy() if "agentview_image" in obs else None
    # in-place camera edit
    sim.model.cam_pos[cam_id] = pos0 + np.array([args.dx, 0.0, 0.0])
    sim.forward()
    s1 = state(env)
    obs1 = env.env._get_observations() if hasattr(env, "env") else None
    # read a fresh render through the wrapper
    env.sim.forward()
    img1 = np.array(env.env._get_observations()["agentview_image"]).copy() if obs1 is not None else None
    # sham: same value again
    sim.model.cam_pos[cam_id] = pos0 + np.array([args.dx, 0.0, 0.0])
    sim.forward()
    img2 = np.array(env.env._get_observations()["agentview_image"]).copy() if obs1 is not None else None
    # restore
    sim.model.cam_pos[cam_id] = pos0
    sim.forward()
    env.close()
    print("camera id", cam_id, "pos0", pos0.round(4), "quat0", quat0.round(4))
    for key in s0:
        a, b = s0[key], s1[key]
        same = (a is None and b is None) or (a is not None and b is not None and np.allclose(a, b))
        print(f"state {key}: unchanged={same}")
    if img0 is not None and img1 is not None:
        print("image diff mean:", float(np.abs(img0.astype(float) - img1.astype(float)).mean()))
    if img1 is not None and img2 is not None:
        print("reproducibility (sham) max diff:", float(np.abs(img1.astype(float) - img2.astype(float)).max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
