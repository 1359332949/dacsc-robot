"""V1.2 pilot: collect paired Clean/Shift DINO features for one task+perturbation.

No policy action, no rollout.  For each init state:
  clean env: set state, settle, render policy views -> DINO patch tokens
  shift env: restore same physical state, render policy views -> DINO patch tokens
Writes per-view feature arrays, raw PNGs, a fixed 30/20 split and a manifest.
"""

from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.state_pairing import compare, digest, model_compatible, restore, snapshot  # noqa: E402
from evaluate_libero import state_from_observation  # noqa: E402
from dacsc.task_pairing import classification_entries, load_task_map, resolve_task_name  # noqa: E402
from sg_cpa_phase4_final import CKPT  # noqa: E402

DUMMY_ACTION = [0.0] * 6 + [-1.0]
SETTLING_STEPS = 10


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combined_image_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*.png")):
        h.update(str(p.relative_to(root)).encode())
        h.update(bytes.fromhex(sha(p)))
    return h.hexdigest()


def configure(args):
    import yaml
    src = args.libero_src.expanduser().resolve()
    root = src / "libero" / "libero"
    if not root.is_dir():
        raise FileNotFoundError(f"LIBERO source not found: {src}")
    sys.path.insert(0, str(src))
    cfg = args.libero_config.expanduser().resolve(); cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "datasets").mkdir(exist_ok=True)
    (cfg / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"), "assets": str(root / "assets"),
        "datasets": str(cfg / "datasets")}))
    import os
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    for name, value in {"int": int, "float": float, "bool": bool, "str": str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def policy_image(raw: np.ndarray, size: int) -> torch.Tensor:
    """Match evaluate_libero: 180-degree rotate, bilinear resize, uint8 NHWC."""
    img = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8)[::-1, ::-1])
    if img.shape[:2] != (size, size):
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.Resampling.BILINEAR))
    return torch.from_numpy(np.ascontiguousarray(img).copy()).unsqueeze(0)  # [1,H,W,3]


def dino_features(policy, raw_by_view: dict[str, np.ndarray], views: list[str], instruction: str):
    size = int(policy.config["image_size"])
    imgs = {v: policy_image(raw_by_view[v], size) for v in views}
    dummy_state = torch.zeros(1, policy.config["state_dim"], device=next(policy.parameters()).device)
    batch = prepare_observation(
        {"images": imgs, "state": dummy_state, "instruction": [instruction]},
        config=policy.config, normalizer=policy.normalizer,
        device=next(policy.parameters()).device)
    with torch.no_grad():
        patches = policy.representation_extractor.visual_encoder(
            torch.cat([batch["images"][v] for v in views], dim=0))
    return {v: patches[i].detach().cpu().numpy() for i, v in enumerate(views)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-suite", default="libero_spatial")
    ap.add_argument("--clean-task-name", required=True)
    ap.add_argument("--clean-bddl", type=Path, required=True)
    ap.add_argument("--clean-init-file", type=Path, required=True)
    ap.add_argument("--clean-language", required=True)
    ap.add_argument("--shift-suite", default="libero_spatial")
    ap.add_argument("--shift-task", type=int, required=True)
    ap.add_argument("--shift-kind", choices=["camera", "background", "lighting"], required=True)
    ap.add_argument("--policy-views", nargs="+", default=["agentview", "eye_in_hand"])
    ap.add_argument("--camera-map", nargs="*", default=["eye_in_hand=robot0_eye_in_hand"])
    ap.add_argument("--target-cameras", nargs="+", default=["agentview"])
    ap.add_argument("--task-map", type=Path, required=True)
    ap.add_argument("--task-classification", type=Path, required=True)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-states", type=int, default=50)
    ap.add_argument("--split-seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    configure(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    camera_map = dict(item.split("=", 1) for item in args.camera_map)
    mapped_views = {v: camera_map.get(v, v) for v in args.policy_views}
    if len(set(mapped_views.values())) != len(mapped_views):
        raise SystemExit("duplicate camera mapping")
    policy = load_policy(CKPT, device=args.device); policy.eval()
    suites = {s: benchmark.get_benchmark_dict()[s]() for s in (args.clean_suite, args.shift_suite)}
    clean_suite = suites[args.clean_suite]; shift_suite = suites[args.shift_suite]
    shift_task = shift_suite.get_task(args.shift_task)
    task_map = load_task_map(args.task_map)
    map_name, _ = resolve_task_name(task_map, args.shift_suite, args.shift_task)
    if map_name != shift_task.name:
        raise RuntimeError(f"task map/benchmark name mismatch: {map_name!r} vs {shift_task.name!r}")
    tc_data = json.loads(args.task_classification.read_text())
    tc_hits = [e for e in classification_entries(tc_data, args.shift_suite, args.shift_task + 1)
               if str(e.get("name", e.get("task_name", ""))) == shift_task.name]
    if len(tc_hits) != 1:
        raise RuntimeError(f"cannot uniquely resolve classification id for {shift_task.name}")
    classification_id = int(tc_hits[0].get("id", args.shift_task + 1))
    shift_bddl = Path(shift_suite.get_task_bddl_file_path(args.shift_task))
    if "_view_" in shift_task.name and not shift_bddl.is_file():
        shift_bddl = Path(get_libero_path("bddl_files")) / shift_task.problem_folder / (shift_task.name.split("_view_")[0] + ".bddl")
    shift_env_bddl = shift_bddl
    if "_view_" in shift_task.name:
        shift_env_bddl = Path(get_libero_path("bddl_files")) / shift_task.problem_folder / (shift_task.name + ".bddl")

    condition = {
        "clean_suite": args.clean_suite, "clean_task_name": args.clean_task_name,
        "shift_suite": args.shift_suite, "domain": args.shift_kind,
        "classification_id": classification_id,
        "variant_name": shift_task.name,
        "benchmark_task_index": int(args.shift_task),
        "resolved_bddl_path": str(shift_bddl),
        "resolved_bddl_sha256": sha(shift_bddl),
        "env_bddl_name": str(shift_env_bddl),
        "classification_category": tc_hits[0].get("category"),
        "classification_difficulty": tc_hits[0].get("difficulty_level"),
    }
    clean_states = torch.load(args.clean_init_file, weights_only=False)
    n = min(args.num_states, len(clean_states))
    perm = np.random.RandomState(args.split_seed).permutation(n)
    dev_idx = sorted(perm[:int(0.6 * n)].tolist())
    held_idx = sorted(perm[int(0.6 * n):].tolist())

    env_c = OffScreenRenderEnv(bddl_file_name=str(args.clean_bddl), camera_heights=256, camera_widths=256,
                               render_gpu_device_id=0)
    env_s = OffScreenRenderEnv(bddl_file_name=str(shift_env_bddl), camera_heights=256, camera_widths=256,
                               render_gpu_device_id=0)
    env_c.reset(); env_s.reset()
    compat = model_compatible(snapshot(env_c), snapshot(env_s), strict_bodies=True,
                              strict_geometry=(args.shift_kind == "camera"),
                              strict_physics=(args.shift_kind == "camera"))
    if not compat["pass"]:
        raise RuntimeError(f"Clean/Shift model incompatible: {compat}")
    features = {c: {v: [] for v in args.policy_views} for c in ("clean", "shift")}
    state8s = []
    records = []
    try:
        for pos, idx in enumerate(range(n)):
            env_c.seed(0); env_c.reset(); env_s.seed(0); env_s.reset()
            env_c.set_init_state(clean_states[idx])
            obs_c = None
            for _ in range(SETTLING_STEPS):
                obs_c, _, done, _ = env_c.step(DUMMY_ACTION)
                if done:
                    raise RuntimeError(f"settling terminated at state {idx}")
            if obs_c is None:
                raise RuntimeError(f"no observation after settling at state {idx}")
            snap = snapshot(env_c)
            restore(env_s, snap)
            cmp = compare(snap, snapshot(env_s))
            if not cmp["all_pass"]:
                records.append({"index": idx, "error": "state_restore_failed", "compare": cmp,
                            "feature_row": None})
                continue
            raw = {"clean": {}, "shift": {}}
            for v in args.policy_views:
                cam = mapped_views[v]
                raw["clean"][v] = env_c.sim.render(camera_name=cam, width=256, height=256)
                raw["shift"][v] = env_s.sim.render(camera_name=cam, width=256, height=256)
                d = args.out / "images" / f"state_{idx:03d}"
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(raw["clean"][v], dtype=np.uint8)).save(d / f"clean_{v}.png")
                Image.fromarray(np.asarray(raw["shift"][v], dtype=np.uint8)).save(d / f"shift_{v}.png")
            feats = dino_features(policy, raw["clean"], args.policy_views, args.clean_language)
            feats_s = dino_features(policy, raw["shift"], args.policy_views, args.clean_language)
            feature_row = len(features["clean"][args.policy_views[0]])
            for v in args.policy_views:
                features["clean"][v].append(feats[v].astype(np.float32))
                features["shift"][v].append(feats_s[v].astype(np.float32))
            state8s.append(state_from_observation(obs_c, policy.normalizer).astype(np.float32))
            records.append({"index": idx, "state_digest": digest(snap), "compare_pass": True,
                            "feature_row": feature_row})
            print(f"[{pos+1}/{n}] state {idx} ok", flush=True)
    finally:
        env_s.close(); env_c.close()

    for c in features:
        for v in features[c]:
            if features[c][v]:
                features[c][v] = np.stack(features[c][v], axis=0)
    np.savez_compressed(args.out / "features.npz",
                        **{f"{c}_{v}": features[c][v] for c in features for v in features[c]},
                        state8=np.stack(state8s, axis=0) if state8s else np.zeros((0, policy.config["state_dim"]), dtype=np.float32))
    manifest = {
        "condition": condition,
        "clean_task_name": args.clean_task_name, "clean_language": args.clean_language,
        "shift_task": args.shift_task, "shift_task_name": shift_task.name,
        "shift_kind": args.shift_kind, "num_states": n,
        "split_seed": args.split_seed, "dev_indices": dev_idx, "heldout_indices": held_idx,
        "policy_views": args.policy_views, "camera_map": mapped_views,
        "target_cameras": args.target_cameras,
        "clean_bddl_sha256": sha(args.clean_bddl), "shift_bddl_sha256": sha(shift_bddl),
        "shift_env_bddl": str(shift_env_bddl),
        "records": records,
        "feature_rows": [r.get("feature_row") for r in records],
        "state8_rows": len(state8s),
        "features": {f"{c}_{v}": list(features[c][v].shape) for c in features for v in features[c]},
        "checkpoint": str(CKPT), "checkpoint_sha256": sha(CKPT),
        "clean_init_file": str(args.clean_init_file), "clean_init_sha256": sha(args.clean_init_file),
        "features_sha256": sha(args.out / "features.npz"),
        "image_hashes_combined": combined_image_hash(args.out / "images"),
        "model_compatible": compat,
        "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
        "capture_script_sha256": sha(Path(__file__)),
        "corp_config_sha256": sha(CKPT.parent / "config.json") if (CKPT.parent / "config.json").is_file() else None,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"num_states": n, "dev": len(dev_idx), "heldout": len(held_idx),
                      "features": manifest["features"]}, indent=2))


if __name__ == "__main__":
    main()
