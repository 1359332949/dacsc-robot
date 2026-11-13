"""V1.1 fail-closed Clean/Shift physical-state pairing audit for LIBERO-Plus.

Preflight checks task identity (language, base task, BDDL goal/objects), model
compatibility, states, target-camera relation, required policy views and real
pixel changes.  Every run writes a structured manifest, including failures.
"""

from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys, time, uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero  # noqa: E402
from dacsc.state_pairing import camera_compare, compare, digest, model_compatible, restore, snapshot  # noqa: E402
from dacsc.task_pairing import task_fingerprint, task_pair_compatible  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def to_hwc(arr) -> tuple[np.ndarray, dict]:
    raw = np.asarray(arr)
    meta = {"raw_dtype": str(raw.dtype), "raw_shape": list(raw.shape),
            "raw_min": float(np.min(raw)) if raw.size else None,
            "raw_max": float(np.max(raw)) if raw.size else None}
    if raw.dtype == np.uint8:
        out = raw
    elif np.issubdtype(raw.dtype, np.floating):
        if not np.isfinite(raw).all() or raw.min() < 0.0 or raw.max() > 1.0:
            raise ValueError(f"unexpected float image range: {meta}")
        out = np.clip(raw * 255.0, 0, 255).round().astype(np.uint8)
    else:
        raise TypeError(f"unsupported image dtype: {raw.dtype}")
    if out.ndim == 3 and out.shape[0] in (1, 3, 4):
        out = out.transpose(1, 2, 0)
    if out.ndim != 3 or out.shape[2] not in (3, 4):
        raise ValueError(f"unexpected image shape after HWC conversion: {out.shape}")
    meta["converted_shape"] = list(out.shape)
    return out, meta


def save_png(arr, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return {"path": str(path), "sha256": sha256_file(path), "shape": list(arr.shape)}


def image_diff(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return {"shape_match": False, "mean_abs_diff": float("inf"), "changed_fraction": 1.0}
    d = np.abs(a - b)
    return {"shape_match": True,
            "mean_abs_diff": float(d.mean() / 255.0),
            "changed_fraction": float((d.max(axis=-1) > 5).mean())}


def load_classification(path: Path | None, task_ids: list[int]) -> dict:
    if path is None:
        return {"provided": False}
    data = json.loads(path.read_text())
    found: dict[int, list] = {tid: [] for tid in task_ids}
    def walk(obj):
        if isinstance(obj, dict):
            for key in ("task_id", "task_index", "index", "id"):
                if key in obj:
                    try:
                        tid = int(obj[key])
                    except Exception:
                        tid = None
                    if tid in found:
                        found[tid].append(obj)
            for v in obj.values(): walk(v)
        elif isinstance(obj, list):
            for v in obj: walk(v)
    walk(data)
    return {"provided": True, "path": str(path), "sha256": sha256_file(path),
            "entries": {str(k): v for k, v in found.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-suite", default="libero_spatial")
    ap.add_argument("--clean-task", type=int, default=0)
    ap.add_argument("--shift-suite", default="libero_spatial")
    ap.add_argument("--shift-task", type=int, default=705)
    ap.add_argument("--init-index", type=int, required=True)
    ap.add_argument("--shift-kind", choices=["background", "lighting", "camera"], required=True)
    ap.add_argument("--policy-views", nargs="+", default=["agentview", "eye_in_hand"])
    ap.add_argument("--target-cameras", nargs="+", default=None,
                    help="cameras that must show the expected visual change; defaults to policy views")
    ap.add_argument("--task-classification", type=Path, default=None)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_plus_config"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/validation/v11_pair_audit"))
    ap.add_argument("--diff-threshold", type=float, default=0.002)
    ap.add_argument("--changed-threshold", type=float, default=0.01)
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:8]
    policy_views = list(args.policy_views)
    target_cameras = list(args.target_cameras or policy_views)
    bad_targets = [c for c in target_cameras if c not in policy_views]
    if bad_targets:
        raise SystemExit(f"target cameras not in policy views: {bad_targets}")

    record: dict = {
        "run_id": run_id,
        "clean": {"suite": args.clean_suite, "task": args.clean_task},
        "shift": {"suite": args.shift_suite, "task": args.shift_task, "kind": args.shift_kind},
        "init_index": args.init_index,
        "policy_views": policy_views,
        "target_cameras": target_cameras,
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "versions": {"python": sys.version, "platform": platform.platform(),
                     "torch": torch.__version__, "numpy": np.__version__},
        "thresholds": {"mean_abs_diff": args.diff_threshold,
                       "changed_fraction": args.changed_threshold},
        "seed": 0, "settling_steps": int(SETTLING_STEPS),
    }
    try:
        import mujoco, robosuite  # noqa: E402
        record["versions"]["mujoco"] = getattr(mujoco, "__version__", None)
        record["versions"]["robosuite"] = getattr(robosuite, "__version__", None)
    except Exception:
        pass
    record["libero_git_rev"] = git_rev(args.libero_src)
    record["libero_src"] = str(args.libero_src)

    class C: pass
    c = C(); c.libero_src = args.libero_src; c.libero_config_path = args.libero_config
    c.render_gpu = args.render_gpu; c.device = args.device
    configure_libero(c)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    env_c = None; env_s = None; exit_code = 0
    try:
        record["task_classification"] = load_classification(
            args.task_classification, [args.clean_task, args.shift_task])
        suites = {s: benchmark.get_benchmark_dict()[s]() for s in (args.clean_suite, args.shift_suite)}
        clean_task = suites[args.clean_suite].get_task(args.clean_task)
        shift_task = suites[args.shift_suite].get_task(args.shift_task)
        clean_bddl = Path(get_libero_path("bddl_files")) / clean_task.problem_folder / clean_task.bddl_file
        shift_bddl = Path(get_libero_path("bddl_files")) / shift_task.problem_folder / shift_task.bddl_file
        clean_init_path = Path(get_libero_path("init_states")) / clean_task.problem_folder / clean_task.init_states_file
        record["clean"].update({"bddl": str(clean_bddl), "bddl_sha256": sha256_file(clean_bddl),
                                "language": clean_task.language})
        record["shift"].update({"bddl": str(shift_bddl), "bddl_sha256": sha256_file(shift_bddl),
                                "language": shift_task.language})
        record["init_file"] = str(clean_init_path)
        record["init_file_sha256"] = sha256_file(clean_init_path)
        clean_fp = task_fingerprint(clean_task, clean_bddl)
        shift_fp = task_fingerprint(shift_task, shift_bddl)
        pairing = task_pair_compatible(clean_fp, shift_fp)
        record["task_fingerprints"] = {"clean": clean_fp, "shift": shift_fp}
        record["task_pairing"] = pairing
        if not pairing["pass"]:
            record["audit"] = {"pass": False, "reason": "task_identity_mismatch"}
            raise SystemExit(2)

        clean_states = torch.load(clean_init_path, weights_only=False)
        n_init = len(clean_states)
        record["n_init"] = n_init
        if not (0 <= args.init_index < n_init):
            record["audit"] = {"pass": False, "reason": "init_index_out_of_range"}
            raise SystemExit(2)
        env_c = OffScreenRenderEnv(bddl_file_name=str(clean_bddl), camera_heights=256,
                                   camera_widths=256, render_gpu_device_id=args.render_gpu)
        try:
            env_s = OffScreenRenderEnv(bddl_file_name=str(shift_bddl), camera_heights=256,
                                       camera_widths=256, render_gpu_device_id=args.render_gpu)
        except Exception:
            env_c.close(); env_c = None
            raise
        for env in (env_c, env_s):
            env.seed(record["seed"]); env.reset()
        env_c.set_init_state(clean_states[args.init_index])
        settling_done = False
        for i in range(SETTLING_STEPS):
            _, _, done, _ = env_c.step(DUMMY_ACTION)
            if done:
                settling_done = True
                record["settling_done_at"] = i
                break
        record["settling_done_early"] = settling_done
        if settling_done:
            record["audit"] = {"pass": False, "reason": "settling_terminated_episode"}
            raise SystemExit(2)

        snap_c = snapshot(env_c); snap_s_pre = snapshot(env_s)
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
        cam = camera_compare(snap_c, snap_s,
                             expect_equal=(args.shift_kind in ("background", "lighting")),
                             target_cameras=target_cameras)
        record.update({"state_digest_clean": digest(snap_c),
                       "state_digest_restored": digest(snap_s),
                       "compare": cmp, "camera": cam})
        missing_views = [v for v in policy_views if v not in snap_c["camera_names"]]
        if missing_views:
            record["audit"] = {"pass": False, "reason": f"policy_views_missing: {missing_views}"}
            raise SystemExit(2)

        run_dir = args.out / args.shift_kind / (
            f"{args.clean_suite}-{args.clean_task}_to_{args.shift_suite}-{args.shift_task}_init{args.init_index}"
        ) / f"run_{run_id}"
        images = {}
        for cam_name in policy_views:
            try:
                arr_c, meta_c = to_hwc(env_c.sim.render(camera_name=cam_name, width=256, height=256))
                arr_s, meta_s = to_hwc(env_s.sim.render(camera_name=cam_name, width=256, height=256))
                ent_c = save_png(arr_c, run_dir / f"clean_{cam_name}.png")
                ent_s = save_png(arr_s, run_dir / f"shift_{cam_name}.png")
                diff = image_diff(arr_c, arr_s)
                images[cam_name] = {"clean": ent_c, "shift": ent_s, "diff": diff,
                                    "image_meta": {"clean": meta_c, "shift": meta_s},
                                    "visual_changed": bool(diff["shape_match"] and
                                                           diff["mean_abs_diff"] >= args.diff_threshold and
                                                           diff["changed_fraction"] >= args.changed_threshold)}
            except Exception as exc:
                images[cam_name] = {"error": repr(exc), "visual_changed": False}
        record["images"] = images
        images_ok = all(name in images and "error" not in images[name] for name in policy_views)
        visual_ok = bool(target_cameras) and all(images.get(c, {}).get("visual_changed", False)
                                                 for c in target_cameras)
        record["images_ok"] = images_ok
        record["visual_shift_ok"] = visual_ok
        audit_ok = bool(pairing["pass"] and compat["pass"] and cmp["all_pass"] and cam["pass"]
                        and images_ok and visual_ok and not settling_done)
        record["audit"] = {"pass": audit_ok, "reason": "ok" if audit_ok else "failed_required_check"}
        exit_code = 0 if audit_ok else 2
    except SystemExit as exc:
        exit_code = int(exc.code or 2)
        record.setdefault("audit", {"pass": False, "reason": "system_exit"})
    except Exception as exc:
        exit_code = 2
        record["unexpected_error"] = repr(exc)
        record["audit"] = {"pass": False, "reason": "unexpected_error"}
    finally:
        if env_s is not None: env_s.close()
        if env_c is not None: env_c.close()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "pair_manifest.jsonl").open("a").write(json.dumps(record, separators=(",", ":")) + "\n")
        (args.out / f"pair_manifest_{run_id}.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({"run_id": run_id, "audit_pass": bool(record.get("audit", {}).get("pass")),
                          "reason": record.get("audit", {}).get("reason"),
                          "exit_code": exit_code}, indent=2))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
