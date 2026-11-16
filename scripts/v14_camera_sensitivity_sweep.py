"""Controlled Camera-parameter sensitivity sweep on frozen CoRP.

Same base task, physical state and instruction; agentview camera is varied over
several controlled parameter deltas.  No policy action, no rollout.
"""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.state_pairing import camera_compare, compare, digest, model_compatible, restore, snapshot  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]
SETTLING_STEPS = 10


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    for name, value in {"int": int, "float": float, "bool": bool, "str": str}.items():
        if name not in np.__dict__: setattr(np, name, value)


def camera_configs():
    return [
        ("base", 0, 0, 100, 0, 0),
        ("h15", 15, 0, 100, 0, 0),
        ("h30", 30, 0, 100, 0, 0),
        ("h46", 46, 0, 100, 0, 0),
        ("h60", 60, 0, 100, 0, 0),
        ("v+10", 0, 10, 100, 0, 0),
        ("v-10", 0, -10, 100, 0, 0),
        ("scale90", 0, 0, 90, 0, 0),
        ("scale110", 0, 0, 110, 0, 0),
    ]


def apply_camera(env, cfg):
    """Set agentview camera exactly as LIBERO's _setup_camera does."""
    from libero.libero.envs.problems.libero_tabletop_manipulation import (
        rotate_around_y, rotate_around_z, scale_distance_from_pivot)
    _, h, v, scale, rot, vv = cfg
    pos = [0.6586131746834771, 0.0, 1.6103500240372423]
    quat = [0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349]
    if int(v) != 0:
        r = rotate_around_y(original_quat=quat, original_pos=pos, degrees=int(v))
        pos, quat = r["new_pos"], r["new_quat"]
    r = rotate_around_z(original_quat=quat, original_pos=pos, degrees=int(h))
    pos, quat = r["new_pos"], r["new_quat"]
    if float(scale) / 100.0 != 1.0:
        r = scale_distance_from_pivot(original_quat=quat, original_pos=pos,
                                      scale_factor=float(scale) / 100.0)
        pos, quat = r["new_pos"], r["new_quat"]
    if int(rot) != 0:
        r = rotate_around_z(original_quat=quat, degrees=int(rot)); quat = r["new_quat"]
    if int(vv) != 0:
        r = rotate_around_y(original_quat=quat, degrees=int(vv)); quat = r["new_quat"]
    names = list(env.sim.model.camera_names); i = names.index("agentview")
    env.sim.model.cam_pos[i] = [round(x, 4) for x in pos]
    env.sim.model.cam_quat[i] = [round(x, 4) for x in quat]
    env.sim.forward()
    return env.sim.model.cam_pos[i].tolist(), env.sim.model.cam_quat[i].tolist()


def policy_image(raw: np.ndarray, size: int) -> torch.Tensor:
    img = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)[::-1, ::-1])
    if img.shape[:2] != (size, size):
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.Resampling.BILINEAR))
    return torch.from_numpy(np.ascontiguousarray(img).copy()).unsqueeze(0)


def dino_features(policy, raw_by_view, views, instruction):
    size = int(policy.config["image_size"]); device = next(policy.parameters()).device
    imgs = {v: policy_image(raw_by_view[v], size).to(device) for v in views}
    batch = prepare_observation({"images": imgs,
                                 "state": torch.zeros(1, policy.config["state_dim"], device=device),
                                 "instruction": [instruction]},
                                config=policy.config, normalizer=policy.normalizer, device=device)
    with torch.no_grad():
        p = policy.representation_extractor.visual_encoder(
            torch.cat([batch["images"][v] for v in views], dim=0))
    return {v: p[i].detach().cpu().numpy() for i, v in enumerate(views)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-bddl", type=Path, required=True)
    ap.add_argument("--clean-init-file", type=Path, required=True)
    ap.add_argument("--clean-language", required=True)
    ap.add_argument("--task-name", required=True)
    ap.add_argument("--policy-views", nargs="+", default=["agentview", "eye_in_hand"])
    ap.add_argument("--camera-map", nargs="*", default=["eye_in_hand=robot0_eye_in_hand"])
    ap.add_argument("--num-states", type=int, default=20)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    configure(args)
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    args.out.mkdir(parents=True, exist_ok=True)
    clean_bddl = args.clean_bddl.expanduser().resolve()
    camera_map = dict(x.split("=", 1) for x in args.camera_map)
    views = list(args.policy_views)
    mapped = {v: camera_map.get(v, v) for v in views}
    configs = camera_configs()
    clean_states = torch.load(args.clean_init_file, weights_only=False)
    n = min(args.num_states, len(clean_states))
    policy = load_policy(CKPT, device=args.device); policy.eval()

    env = OffScreenRenderEnv(bddl_file_name=str(clean_bddl), camera_heights=256,
                             camera_widths=256, render_gpu_device_id=0)
    try:
        feats = {name: {v: [] for v in views} for name, *_ in configs}
        camera_meta = {}
        records = []
        for idx in range(n):
            env.seed(0); env.reset(); env.set_init_state(clean_states[idx])
            for _ in range(SETTLING_STEPS):
                _, _, done, _ = env.step(DUMMY_ACTION)
                if done: raise RuntimeError(f"settling done at state {idx}")
            snap = snapshot(env)
            for cfg in configs:
                name = cfg[0]
                cam_pos, cam_quat = apply_camera(env, cfg)
                raw = {v: env.sim.render(camera_name=mapped[v], width=256, height=256) for v in views}
                d = args.out / "images" / f"state_{idx:03d}" / name
                d.mkdir(parents=True, exist_ok=True)
                for v in views:
                    Image.fromarray(np.asarray(raw[v], dtype=np.uint8)).save(d / f"{v}.png")
                f = dino_features(policy, raw, views, args.clean_language)
                for v in views: feats[name][v].append(f[v].astype(np.float32))
                camera_meta[name] = {"cam_pos": cam_pos, "cam_quat": cam_quat}
            records.append({"state": idx, "snapshot_digest": digest(snap)})
            print(f"[{idx+1}/{n}] ok", flush=True)
        arrays = {}
        for name, *_ in configs:
            for v in views:
                arrays[f"{name}_{v}"] = np.stack(feats[name][v], axis=0)
        np.savez_compressed(args.out / "features.npz", **arrays)
        manifest = {
            "task_name": args.task_name, "num_states": n,
            "camera_configs": [dict(name=c[0], horizon=c[1], vertical=c[2],
                                    scale=c[3], end_point_rot=c[4], end_point_vertical=c[5])
                               for c in configs],
            "policy_views": views, "camera_map": mapped,
            "clean_init_file": str(args.clean_init_file), "clean_init_sha256": sha(args.clean_init_file),
            "clean_bddl": str(args.clean_bddl), "clean_bddl_sha256": sha(args.clean_bddl),
            "checkpoint": str(CKPT), "checkpoint_sha256": sha(CKPT),
            "features_sha256": sha(args.out / "features.npz"),
            "records": records, "camera_meta": camera_meta,
            "features": {k: list(v.shape) for k, v in arrays.items()},
        }
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({"num_states": n, "configs": len(configs), "features": manifest["features"]}, indent=2))
    finally:
        try: env.close()
        except Exception: pass


if __name__ == "__main__":
    main()
