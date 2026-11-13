"""V1.1 fail-closed Clean/Shift physical-state pairing audit for LIBERO-Plus.

Success requires all of:
  physical state match, model compatibility, expected camera relation,
  required views captured, and an actual measured visual difference.
Any failed required check exits non-zero.
"""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero  # noqa: E402
from dacsc.state_pairing import camera_compare, compare, digest, model_compatible, restore, snapshot  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def to_hwc(arr):
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = arr.transpose(1, 2, 0)
    return arr


def save_png(arr, path: Path) -> dict:
    arr = to_hwc(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return {"path": str(path), "sha256": file_sha256(path), "shape": list(arr.shape)}


def image_diff(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"shape_match": False, "mean_abs_diff": float("inf"), "changed_fraction": 1.0}
    d = np.abs(a - b)
    return {"shape_match": True,
            "mean_abs_diff": float(d.mean() / 255.0),
            "changed_fraction": float((d.max(axis=-1) > 5).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-suite", default="libero_spatial")
    ap.add_argument("--clean-task", type=int, default=0)
    ap.add_argument("--shift-suite", default="libero_spatial")
    ap.add_argument("--shift-task", type=int, default=705)
    ap.add_argument("--init-index", type=int, required=True)
    ap.add_argument("--shift-kind", choices=["background", "lighting", "camera"], required=True)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_plus_config"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-language-diff", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/validation/v11_pair_audit"))
    ap.add_argument("--diff-threshold", type=float, default=0.002)
    ap.add_argument("--changed-threshold", type=float, default=0.01)
    args = ap.parse_args()

    class C: pass
    c = C(); c.libero_src = args.libero_src; c.libero_config_path = args.libero_config
    c.render_gpu = 0; c.device = args.device
    configure_libero(c)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suites = {s: benchmark.get_benchmark_dict()[s]() for s in (args.clean_suite, args.shift_suite)}
    clean_task = suites[args.clean_suite].get_task(args.clean_task)
    shift_task = suites[args.shift_suite].get_task(args.shift_task)
    clean_bddl = Path(get_libero_path("bddl_files")) / clean_task.problem_folder / clean_task.bddl_file
    shift_bddl = Path(get_libero_path("bddl_files")) / shift_task.problem_folder / shift_task.bddl_file
    clean_init_path = Path(get_libero_path("init_states")) / clean_task.problem_folder / clean_task.init_states_file
    clean_states = torch.load(clean_init_path, weights_only=False)
    n_init = len(clean_states)
    if not (0 <= args.init_index < n_init):
        raise SystemExit(f"init-index {args.init_index} out of range [0, {n_init})")

    language_ok = clean_task.language == shift_task.language
    if not language_ok and not args.allow_language_diff:
        raise SystemExit(f"task language mismatch: {clean_task.language!r} vs {shift_task.language!r}")

    env_c = None; env_s = None
    record: dict = {
        "clean": {"suite": args.clean_suite, "task": args.clean_task, "bddl": str(clean_bddl),
                  "language": clean_task.language},
        "shift": {"suite": args.shift_suite, "task": args.shift_task, "bddl": str(shift_bddl),
                  "language": shift_task.language, "kind": args.shift_kind},
        "init_index": args.init_index,
        "n_init": n_init,
        "init_file": str(clean_init_path),
        "init_file_sha256": file_sha256(clean_init_path),
        "language_ok": bool(language_ok),
        "out": str(args.out),
    }
    try:
        env_c = OffScreenRenderEnv(bddl_file_name=str(clean_bddl), camera_heights=256,
                                   camera_widths=256, render_gpu_device_id=0)
        try:
            env_s = OffScreenRenderEnv(bddl_file_name=str(shift_bddl), camera_heights=256,
                                       camera_widths=256, render_gpu_device_id=0)
        except Exception:
            env_c.close(); env_c = None
            raise
        for env in (env_c, env_s):
            env.seed(0); env.reset()
        obs = env_c.set_init_state(clean_states[args.init_index])
        for _ in range(SETTLING_STEPS):
            obs, _, done, _ = env_c.step(DUMMY_ACTION)
            if done: break
        snap_c = snapshot(env_c)
        snap_s_pre = snapshot(env_s)
        compat = model_compatible(snap_c, snap_s_pre, strict_bodies=True)
        record["model_compatible"] = compat
        if not compat["pass"]:
            record["audit"] = {"pass": False, "reason": "model_incompatible"}
            raise SystemExit(2)
        try:
            restore(env_s, snap_c)
        except Exception as exc:
            record["restore_error"] = repr(exc)
            record["audit"] = {"pass": False, "reason": "restore_failed"}
            raise SystemExit(3)
        snap_s = snapshot(env_s)
        cmp = compare(snap_c, snap_s)
        cam = camera_compare(snap_c, snap_s, expect_equal=(args.shift_kind in ("background", "lighting")))
        record.update({"state_digest_clean": digest(snap_c),
                       "state_digest_restored": digest(snap_s),
                       "compare": cmp, "camera": cam})

        run_dir = args.out / args.shift_kind / (
            f"{args.clean_suite}-{args.clean_task}_to_{args.shift_suite}-{args.shift_task}_init{args.init_index}")
        images = {}
        for cam_name in snap_c["camera_names"]:
            try:
                arr_c = to_hwc(env_c.sim.render(camera_name=cam_name, width=256, height=256))
                arr_s = to_hwc(env_s.sim.render(camera_name=cam_name, width=256, height=256))
                ent_c = save_png(arr_c, run_dir / f"clean_{cam_name}.png")
                ent_s = save_png(arr_s, run_dir / f"shift_{cam_name}.png")
                diff = image_diff(arr_c, arr_s)
                images[cam_name] = {"clean": ent_c, "shift": ent_s, "diff": diff,
                                    "visual_changed": bool(diff["shape_match"] and
                                                           diff["mean_abs_diff"] >= args.diff_threshold and
                                                           diff["changed_fraction"] >= args.changed_threshold)}
            except Exception as exc:
                images[cam_name] = {"error": repr(exc), "visual_changed": False}
        record["images"] = images
        required_views = list(snap_c["camera_names"])
        images_ok = all(name in images and "error" not in images[name] for name in required_views)
        visual_ok = bool(required_views) and all(images[name]["visual_changed"] for name in required_views)
        record["required_views"] = required_views
        record["images_ok"] = images_ok
        record["visual_shift_ok"] = visual_ok
        audit_ok = bool(language_ok and compat["pass"] and cmp["all_pass"] and cam["pass"] and images_ok and visual_ok)
        record["audit"] = {"pass": audit_ok,
                           "reason": "ok" if audit_ok else "failed_required_check"}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "pair_manifest.jsonl").write_text(json.dumps(record, separators=(",", ":")) + "\n")
        (args.out / "pair_manifest.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"audit_pass": audit_ok,
                          "state_pass": cmp["all_pass"],
                          "model_pass": compat["pass"],
                          "camera_pass": cam["pass"],
                          "images_ok": images_ok,
                          "visual_shift_ok": visual_ok,
                          "language_ok": language_ok}, indent=2))
        if not audit_ok:
            raise SystemExit(2)
    finally:
        if env_s is not None: env_s.close()
        if env_c is not None: env_c.close()


if __name__ == "__main__":
    main()
