"""No policy. Dump compiled cameras/lights/scene and first-frame evidence at fixed physical state."""

import argparse, ctypes, hashlib, json, random
from pathlib import Path
import numpy as np, torch
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

p = argparse.ArgumentParser()
p.add_argument("tag")
p.add_argument("out", type=Path)
a = p.parse_args()
task = json.load(open("/hy-tmp/patch-diag/runs/smolvla_reproduction_v1/tasks.json"))[0]
base = Path(get_libero_path("bddl_files")) / "libero_spatial" / (task + ".bddl")
init = torch.load(
    Path(get_libero_path("init_states")) / "libero_spatial" / (task + ".pruned_init"),
    map_location="cpu",
    weights_only=False,
)[0]
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
ctypes.CDLL(None).srand(ctypes.c_uint(0))
env = OffScreenRenderEnv(
    bddl_file_name=str(base),
    camera_heights=256,
    camera_widths=256,
    render_gpu_device_id=0,
)
try:
    env.noise = 0
    env.seed(0)
    env.reset()
    obs = env.set_init_state(init)
    for _ in range(10):
        obs = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))[0]
    m = env.sim.model
    data = {
        k: np.asarray(getattr(m, k)).copy()
        for k in [
            "cam_pos",
            "cam_quat",
            "cam_fovy",
            "light_pos",
            "light_dir",
            "light_diffuse",
            "light_ambient",
            "light_specular",
            "light_castshadow",
            "geom_rgba",
            "geom_size",
            "geom_pos",
            "geom_quat",
            "mat_rgba",
            "tex_data",
            "tex_adr",
            "tex_height",
            "tex_width",
            "tex_nchannel",
            "mat_texid",
            "body_mass",
            "geom_friction",
        ]
        if hasattr(m, k)
    }
    data.update(
        qpos=env.sim.data.qpos.copy(),
        qvel=env.sim.data.qvel.copy(),
        agent=obs["agentview_image"],
        wrist=obs["robot0_eye_in_hand_image"],
    )
    a.out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(a.out / (a.tag + ".npz"), **data)
    (a.out / (a.tag + ".xml")).write_text(env.sim.model.get_xml())
finally:
    env.close()
