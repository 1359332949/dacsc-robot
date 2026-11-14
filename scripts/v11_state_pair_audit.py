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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from dacsc.state_pairing import camera_compare, compare, digest, model_compatible, restore, snapshot  # noqa: E402
from dacsc.task_pairing import (canonical_base_name, canonical_instruction,  # noqa: E402
                                classification_entries, load_task_map, resolve_task_name,
                                task_fingerprint, task_pair_compatible, validate_classification)
from dacsc.pairing_checks import freeze_policy_instruction, validate_clean_file_identity  # noqa: E402


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
    ap.add_argument("--camera-map", nargs="*", default=["eye_in_hand=robot0_eye_in_hand"],
                    help="policy-view=MuJoCo-camera mapping")
    ap.add_argument("--target-cameras", nargs="+", default=None,
                    help="policy views that must show the expected visual change")
    ap.add_argument("--task-map", type=Path, required=True,
                    help="libero_suite_task_map.py (or JSON) from the installed revision")
    ap.add_argument("--task-classification", type=Path, required=True,
                    help="task_classification.json from the installed revision")
    ap.add_argument("--clean-source", choices=["plus", "vanilla"], default="plus")
    ap.add_argument("--clean-task-name", default=None,
                    help="explicit vanilla base task name; required for --clean-source vanilla")
    ap.add_argument("--clean-bddl", type=Path, default=None)
    ap.add_argument("--clean-language", default=None)
    ap.add_argument("--clean-init-file", type=Path, default=None)
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
    camera_map = {}
    for item in args.camera_map:
        if "=" not in item:
            raise SystemExit(f"bad --camera-map entry: {item!r}")
        k, v = item.split("=", 1); camera_map[k] = v
    if args.shift_kind == "camera" and not args.target_cameras:
        raise SystemExit("Camera mode requires explicit --target-cameras")
    unknown_map = [k for k in camera_map if k not in policy_views]
    if unknown_map:
        raise SystemExit(f"camera-map keys not in policy views: {unknown_map}")
    mapped_all = {v: camera_map.get(v, v) for v in policy_views}
    if len(set(mapped_all.values())) != len(policy_views):
        raise SystemExit(f"duplicate MuJoCo camera mapping: {mapped_all}")
    target_cameras = list(args.target_cameras or policy_views)
    bad_targets = [c for c in target_cameras if c not in policy_views]
    if bad_targets:
        raise SystemExit(f"target cameras not in policy views: {bad_targets}")
    target_camera_names = [camera_map.get(c, c) for c in target_cameras]

    record: dict = {
        "run_id": run_id,
        "clean": {"suite": args.clean_suite,
                  "task": args.clean_task if args.clean_source == "plus" else None,
                  "source": args.clean_source,
                  "task_name": args.clean_task_name},
        "shift": {"suite": args.shift_suite, "task": args.shift_task, "kind": args.shift_kind},
        "init_index": args.init_index,
        "policy_views": policy_views,
        "camera_map": camera_map,
        "target_cameras": target_cameras,
        "target_camera_names": target_camera_names,
        "cli": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "versions": {"python": sys.version, "platform": platform.platform(),
                     "torch": torch.__version__, "numpy": np.__version__},
        "thresholds": {"mean_abs_diff": args.diff_threshold,
                       "changed_fraction": args.changed_threshold},
        "seed": 0, "settling_steps": None,
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
    env_c = None; env_s = None; exit_code = 0
    try:
        from evaluate_libero import (DUMMY_ACTION as _DUMMY_ACTION,
                                     SETTLING_STEPS as _SETTLING_STEPS,
                                     configure_libero)
        record["settling_steps"] = int(_SETTLING_STEPS)
        configure_libero(c)
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        tc_data = json.loads(args.task_classification.read_text())
        record["task_classification"] = {
            "path": str(args.task_classification), "sha256": sha256_file(args.task_classification)}
        task_map = load_task_map(args.task_map)
        record["task_map"] = {"path": str(args.task_map), "sha256": sha256_file(args.task_map),
                              "n_suites": len(task_map)}
        clean_map_name, clean_map_base = (
            (args.clean_task_name, canonical_base_name(args.clean_task_name))
            if args.clean_source == "vanilla" else
            resolve_task_name(task_map, args.clean_suite, args.clean_task))
        shift_map_name, shift_map_base = resolve_task_name(task_map, args.shift_suite, args.shift_task)
        record["task_map_names"] = {"clean": clean_map_name, "shift": shift_map_name,
                                    "clean_base": clean_map_base, "shift_base": shift_map_base}
        if clean_map_base != shift_map_base:
            record["audit"] = {"pass": False, "reason": "task_map_base_mismatch"}
            raise SystemExit(2)
        if args.clean_source == "vanilla" and not all(
                [args.clean_task_name, args.clean_bddl, args.clean_language, args.clean_init_file]):
            record["audit"] = {"pass": False, "reason": "vanilla_clean_requires_name_bddl_language_init"}
            raise SystemExit(2)
        expected_categories = {
            "camera": {"camera viewpoints"},
            "background": {"background textures"},
            "lighting": {"light conditions"},
        }[args.shift_kind]
        expected = expected_categories
        clean_entries = (classification_entries(tc_data, args.clean_suite, args.clean_task)
                         if args.clean_source == "plus" else [])
        shift_entries = classification_entries(tc_data, args.shift_suite, args.shift_task)
        clean_cls = (validate_classification(clean_entries, expected_name=clean_map_name)
                     if args.clean_source == "plus"
                     else {"checks": {"present": True}, "pass": True, "categories": []})
        shift_cls = validate_classification(shift_entries, expected, expected_name=shift_map_name)
        record["classification"] = {"clean": clean_cls, "shift": shift_cls,
                                    "clean_entries": clean_entries, "shift_entries": shift_entries}
        if not clean_cls["pass"] or not shift_cls["pass"]:
            record["audit"] = {"pass": False, "reason": "task_classification_mismatch"}
            raise SystemExit(2)
        # A Clean reference must not itself be a labeled visual/robot/language variant.
        clean_cats = [c.lower() for c in clean_cls.get("categories", []) if c]
        variant_tokens = ("camera", "background", "light", "table", "language", "layout", "robot", "sensor", "noise")
        if any(any(x in c for x in variant_tokens) for c in clean_cats):
            record["audit"] = {"pass": False, "reason": "clean_reference_is_variant"}
            raise SystemExit(2)
        suites = {s: benchmark.get_benchmark_dict()[s]() for s in (args.clean_suite, args.shift_suite)}
        clean_suite_obj = suites[args.clean_suite]
        shift_suite_obj = suites[args.shift_suite]
        if args.clean_source == "vanilla":
            from types import SimpleNamespace
            clean_task = SimpleNamespace(name=args.clean_task_name, language=args.clean_language,
                                         problem_folder=args.clean_suite,
                                         bddl_file=args.clean_bddl.name,
                                         init_states_file=args.clean_init_file.name)
            clean_bddl = args.clean_bddl
            clean_init_path = args.clean_init_file
            clean_states = torch.load(clean_init_path, weights_only=False)
            record["init_state_source"] = "explicit_vanilla_file"
        else:
            clean_task = clean_suite_obj.get_task(args.clean_task)
            clean_bddl = Path(get_libero_path("bddl_files")) / clean_task.problem_folder / clean_task.bddl_file
            clean_init_path = Path(get_libero_path("init_states")) / clean_task.problem_folder / clean_task.init_states_file
            clean_states = clean_suite_obj.get_task_init_states(args.clean_task)
            record["init_state_source"] = "clean_suite.get_task_init_states"
        shift_task = shift_suite_obj.get_task(args.shift_task)
        clean_id = validate_clean_file_identity(clean_map_name, clean_bddl, clean_init_path)
        record["clean_file_identity"] = clean_id
        if not clean_id["pass"]:
            record["audit"] = {"pass": False, "reason": "clean_file_identity_mismatch"}
            raise SystemExit(2)
        clean_policy = args.clean_language if args.clean_source == "vanilla" else clean_task.language
        prompt = freeze_policy_instruction(
            clean_policy, shift_task.language,
            clean_task_name=clean_map_name if args.clean_source == "vanilla" else None,
            require_vanilla_source=(args.clean_source == "vanilla"))
        record["policy_instruction"] = {
            "clean_raw": clean_task.language, "shift_raw": shift_task.language,
            "clean_policy_instruction": prompt["clean_policy_instruction"],
            "shift_policy_instruction": prompt["shift_policy_instruction"],
            "expected_vanilla_instruction": prompt.get("expected_vanilla_instruction"),
            "checks": prompt["checks"],
        }
        if not prompt["pass"]:
            record["audit"] = {"pass": False, "reason": "policy_instruction_mismatch"}
            raise SystemExit(2)
        if args.clean_source == "vanilla":
            clean_bddl = args.clean_bddl.expanduser().resolve()
            clean_init_path = args.clean_init_file.expanduser().resolve()
            if not clean_bddl.is_file():
                raise FileNotFoundError(clean_bddl)
            if not clean_init_path.is_file():
                raise FileNotFoundError(clean_init_path)
        shift_bddl = Path(get_libero_path("bddl_files")) / shift_task.problem_folder / shift_task.bddl_file
        record["clean"].update({"bddl_input": str(args.clean_bddl) if args.clean_bddl else None,
                                "bddl": str(clean_bddl), "bddl_sha256": sha256_file(clean_bddl),
                                "language": clean_task.language,
                                "policy_instruction": args.clean_language if args.clean_source == "vanilla" else clean_task.language})
        record["shift"].update({"bddl": str(shift_bddl), "bddl_sha256": sha256_file(shift_bddl),
                                "language": shift_task.language,
                                "canonical_language": canonical_instruction(shift_task.language)})
        record["init_file"] = str(clean_init_path)
        record["init_file_sha256"] = sha256_file(clean_init_path)
        clean_fp = task_fingerprint(clean_task, clean_bddl)
        shift_fp = task_fingerprint(shift_task, shift_bddl)
        pairing = task_pair_compatible(clean_fp, shift_fp)
        record["task_fingerprints"] = {"clean": clean_fp, "shift": shift_fp}
        record["task_pairing"] = pairing
        # Three-way identity: task map name == benchmark task name; classification name if present.
        shift_cls_names = [n for n in shift_cls.get("names", []) if n]
        consistency = {
            "clean_task_vs_bddl": bool(clean_id["checks"]["bddl_stem"]),
            "clean_task_vs_init": bool(clean_id["checks"]["init_name"]),
            "shift_map_vs_benchmark": bool(shift_map_name == shift_task.name),
            "shift_class_unique": bool(len(shift_entries) == 1),
            "shift_class_name_full": bool(shift_cls_names) and any(
                (n[:-5] if n.endswith(".bddl") else n) == shift_map_name for n in shift_cls_names),
        }
        record["identity_consistency"] = consistency
        if not all(consistency.values()):
            record["audit"] = {"pass": False, "reason": "identity_consistency_failed"}
            raise SystemExit(2)
        if not pairing["pass"]:
            record["audit"] = {"pass": False, "reason": "task_identity_mismatch"}
            raise SystemExit(2)

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
        for i in range(_SETTLING_STEPS):
            _, _, done, _ = env_c.step(_DUMMY_ACTION)
            if done:
                settling_done = True
                record["settling_done_at"] = i
                break
        record["settling_done_early"] = settling_done
        if settling_done:
            record["audit"] = {"pass": False, "reason": "settling_terminated_episode"}
            raise SystemExit(2)

        snap_c = snapshot(env_c); snap_s_pre = snapshot(env_s)
        compat = model_compatible(snap_c, snap_s_pre, strict_bodies=True,
                                  strict_geometry=(args.shift_kind == "camera"),
                                  strict_physics=(args.shift_kind == "camera"))
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
                             target_cameras=target_camera_names)
        record.update({"state_digest_clean": digest(snap_c),
                       "state_digest_restored": digest(snap_s),
                       "compare": cmp, "camera": cam})
        mapped_views = {v: camera_map.get(v, v) for v in policy_views}
        missing_views = [v for v, c in mapped_views.items() if c not in snap_c["camera_names"]]
        if missing_views:
            record["audit"] = {"pass": False, "reason": f"policy_views_missing: {missing_views}"}
            raise SystemExit(2)

        clean_label = (args.clean_task_name if args.clean_source == "vanilla"
                       else str(args.clean_task))
        run_dir = args.out / args.shift_kind / (
            f"{args.clean_suite}-{clean_label}_to_{args.shift_suite}-{args.shift_task}_init{args.init_index}"
        ) / f"run_{run_id}"
        images = {}
        for view_name in policy_views:
            cam_name = mapped_views[view_name]
            try:
                arr_c, meta_c = to_hwc(env_c.sim.render(camera_name=cam_name, width=256, height=256))
                arr_s, meta_s = to_hwc(env_s.sim.render(camera_name=cam_name, width=256, height=256))
                ent_c = save_png(arr_c, run_dir / f"clean_{view_name}.png")
                ent_s = save_png(arr_s, run_dir / f"shift_{view_name}.png")
                diff = image_diff(arr_c, arr_s)
                images[view_name] = {"camera_name": cam_name, "clean": ent_c, "shift": ent_s, "diff": diff,
                                    "image_meta": {"clean": meta_c, "shift": meta_s},
                                    "visual_changed": bool(diff["shape_match"] and
                                                           diff["mean_abs_diff"] >= args.diff_threshold and
                                                           diff["changed_fraction"] >= args.changed_threshold)}
            except Exception as exc:
                images[view_name] = {"camera_name": cam_name, "error": repr(exc), "visual_changed": False}
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
        cleanup_errors = []
        for label, env in (("shift", env_s), ("clean", env_c)):
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    cleanup_errors.append({"env": label, "error": repr(exc)})
        record["cleanup_errors"] = cleanup_errors
        try:
            args.out.mkdir(parents=True, exist_ok=True)
            with (args.out / "pair_manifest.jsonl").open("a") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            (args.out / f"pair_manifest_{run_id}.json").write_text(json.dumps(record, indent=2))
        except Exception as exc:
            print(f"manifest write failed: {exc!r}", file=sys.stderr)
        print(json.dumps({"run_id": run_id, "audit_pass": bool(record.get("audit", {}).get("pass")),
                          "reason": record.get("audit", {}).get("reason"),
                          "exit_code": exit_code, "cleanup_errors": cleanup_errors}, indent=2))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
