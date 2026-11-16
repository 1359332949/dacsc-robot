"""Phase-A audit: construct one cell per domain and verify pairing.

No policy, no adaptation, no rollout.  Checks state equality, camera/noise
application, language consistency, and target/control-view image differences.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts")); sys.path.insert(0, str(ROOT / "repos/CoRP"))
from dacsc.state_pairing import camera_compare, compare, digest, model_compatible, snapshot  # noqa: E402
from dacsc.libero_camera import apply_camera, parse_view_name  # noqa: E402
from dacsc.libero_noise import apply_agentview_noise  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]
SETTLING_STEPS = 10
OBS_KEYS = {"agentview": "agentview_image", "eye_in_hand": "robot0_eye_in_hand_image"}


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


def image_diff(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"shape_match": False, "mean_abs_diff": float("inf")}
    return {"shape_match": True, "mean_abs_diff": float(np.abs(a - b).mean() / 255.0),
            "changed_fraction": float((np.abs(a - b).max(axis=-1) > 5).mean())}


def find_variant(matrix, suite, domain, cid):
    for b in matrix["base_tasks"]:
        if b["suite"] != suite: continue
        for v in b["variants"][domain]:
            if int(v["classification_id"]) == int(cid):
                return b, v
    raise KeyError(f"cell not found: {suite}/{domain}/{cid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", required=True, help="domain:id, e.g. background:138")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--num-states", type=int, default=3)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    configure(args)
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    matrix = json.loads(args.matrix.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in args.cells:
        domain, cid = spec.split(":")
        b, v = find_variant(matrix, args.suite, domain, cid)
        base_states = torch.load(b["state_split"]["source_file"], weights_only=False)
        idxs = b["state_split"]["dev_indices"][:args.num_states]
        clean_env = OffScreenRenderEnv(bddl_file_name=v["base_bddl"], camera_heights=256,
                                       camera_widths=256, render_gpu_device_id=0)
        variant_env = None
        if domain in ("background", "lighting"):
            variant_env = OffScreenRenderEnv(bddl_file_name=v["env_bddl_name"], camera_heights=256,
                                             camera_widths=256, render_gpu_device_id=0)
        elif domain == "sensor":
            variant_env = OffScreenRenderEnv(bddl_file_name=v["env_bddl_name"], camera_heights=256,
                                             camera_widths=256, render_gpu_device_id=0)
        rec = {"domain": domain, "classification_id": cid, "name": v["name"],
               "variant_kind": v["variant_kind"], "base_bddl": v["base_bddl"],
               "states": []}
        try:
            for idx in idxs:
                clean_env.seed(0); clean_env.reset(); clean_env.set_init_state(base_states[idx])
                for _ in range(SETTLING_STEPS):
                    _, _, done, _ = clean_env.step(DUMMY_ACTION)
                    if done: raise RuntimeError("settling done")
                clean_obs = clean_env.regenerate_obs_from_state(base_states[idx])
                snap_c = snapshot(clean_env)
                st = {"state": idx}
                if domain in ("background", "lighting"):
                    variant_env.seed(0); variant_env.reset(); variant_env.set_init_state(base_states[idx])
                    var_obs = variant_env.regenerate_obs_from_state(base_states[idx])
                    snap_v = snapshot(variant_env)
                    st["state_compare"] = compare(snap_c, snap_v)
                    st["camera_compare"] = camera_compare(snap_c, snap_v, expect_equal=True)
                    st["model_compatible"] = model_compatible(
                        snap_c, snap_v, strict_bodies=True,
                        strict_geometry=True, strict_physics=False)
                    st["language_equal"] = str(getattr(clean_env, "language_instruction", "")) == str(getattr(variant_env, "language_instruction", ""))
                elif domain == "sensor":
                    variant_env.seed(0); variant_env.reset()
                    raw_v = variant_env.regenerate_obs_from_state(base_states[idx])
                    var_obs, meta = apply_agentview_noise(raw_v, variant_env.noise, seed=1234 + idx)
                    snap_v = snapshot(variant_env)
                    st["noise"] = meta
                    st["state_compare"] = compare(snap_c, snap_v)
                    st["camera_compare"] = camera_compare(snap_c, snap_v, expect_equal=True)
                    st["model_compatible"] = model_compatible(
                        snap_c, snap_v, strict_bodies=True,
                        strict_geometry=True, strict_physics=False)
                    st["language_equal"] = str(getattr(clean_env, "language_instruction", "")) == str(getattr(variant_env, "language_instruction", ""))
                else:  # camera
                    cfg = parse_view_name(v["name"])
                    clean_cam = dict(zip(snap_c["camera_names"],
                                         zip(snap_c["cam_pos"], snap_c["cam_quat"])))
                    cam_pos, cam_quat = apply_camera(clean_env, cfg)
                    var_obs = clean_env.regenerate_obs_from_state(base_states[idx])
                    snap_v = snapshot(clean_env)
                    var_cam = dict(zip(snap_v["camera_names"],
                                       zip(snap_v["cam_pos"], snap_v["cam_quat"])))
                    st["camera_config"] = cfg
                    st["camera_changed"] = (clean_cam["agentview"] != var_cam["agentview"])
                    st["non_target_camera_unchanged"] = all(
                        clean_cam[c] == var_cam[c] for c in clean_cam if c != "agentview")
                    st["camera_pos_quat"] = {"pos": cam_pos, "quat": cam_quat}
                    st["state_compare"] = compare(snap_c, snap_v)
                    st["state_compare"]["checks"].pop("camera_names", None)
                    st["model_compatible"] = model_compatible(
                        snap_c, snap_v, strict_bodies=True,
                        strict_geometry=True, strict_physics=True)
                    st["language_equal"] = True
                for view, key in OBS_KEYS.items():
                    if key in clean_obs and key in var_obs:
                        st[f"image_{view}"] = image_diff(clean_obs[key], var_obs[key])
                rec["states"].append(st)
                print(json.dumps({"domain": domain, "state": idx,
                                  "image_agent": st.get("image_agentview")}, ensure_ascii=False), flush=True)
        except Exception as exc:
            rec["error"] = repr(exc)
        finally:
            if variant_env is not None: variant_env.close()
            clean_env.close()
        rec["all_states_state_pass"] = all(
            (x.get("state_compare") or {}).get("all_pass", False) for x in rec["states"])
        rec["all_states_language"] = all(x.get("language_equal", False) for x in rec["states"])
        results.append(rec)
    (args.out / "phase_a_cell_audit.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"cells": len(results), "errors": sum(1 for r in results if "error" in r)}, indent=2))


if __name__ == "__main__":
    main()
