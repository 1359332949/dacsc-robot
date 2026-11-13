"""V1.1: audit same-physical-state Clean/Shift pairing in LIBERO-Plus."""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero  # noqa: E402
from dacsc.state_pairing import camera_equal, compare, digest, restore, snapshot  # noqa: E402


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def save_jpg(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path, quality=92)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-suite", default="libero_spatial")
    ap.add_argument("--clean-task", type=int, default=0)
    ap.add_argument("--shift-suite", default="libero_spatial")
    ap.add_argument("--shift-task", type=int, default=705)
    ap.add_argument("--init-index", type=int, default=0)
    ap.add_argument("--shift-kind", choices=["background", "lighting", "camera"], default="camera")
    ap.add_argument("--out", type=Path, default=Path("/hy-tmp/validation/v11_pair_audit"))
    args = ap.parse_args()

    class C: pass
    c = C(); c.libero_src = Path("/hy-tmp/libero_plus"); c.libero_config_path = Path("/hy-tmp/libero_plus_config")
    c.render_gpu = 0; c.device = "cuda:0"
    configure_libero(c)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suites = {s: benchmark.get_benchmark_dict()[s]() for s in (args.clean_suite, args.shift_suite)}
    clean_task = suites[args.clean_suite].get_task(args.clean_task)
    shift_task = suites[args.shift_suite].get_task(args.shift_task)
    clean_bddl = Path(get_libero_path("bddl_files")) / clean_task.problem_folder / clean_task.bddl_file
    shift_bddl = Path(get_libero_path("bddl_files")) / shift_task.problem_folder / shift_task.bddl_file
    clean_states = torch.load(Path(get_libero_path("init_states")) / clean_task.problem_folder / clean_task.init_states_file,
                              weights_only=False)

    env_c = OffScreenRenderEnv(bddl_file_name=str(clean_bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    env_s = OffScreenRenderEnv(bddl_file_name=str(shift_bddl), camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    try:
        for env in (env_c, env_s):
            env.seed(0); env.reset()
        obs = env_c.set_init_state(clean_states[args.init_index % len(clean_states)])
        for _ in range(SETTLING_STEPS):
            obs, _, done, _ = env_c.step(DUMMY_ACTION)
            if done: break
        snap_c = snapshot(env_c)
        restore(env_s, snap_c)
        snap_s = snapshot(env_s)
        cmp = compare(snap_c, snap_s)
        cam_eq = camera_equal(snap_c, snap_s)
        expected_cam_equal = args.shift_kind in ("background", "lighting")
        images = {}
        for cam in snap_c["camera_names"]:
            try:
                arr_c = env_c.sim.render(camera_name=cam, width=256, height=256)
                arr_s = env_s.sim.render(camera_name=cam, width=256, height=256)
                pc = args.out / args.shift_kind / f"clean_{cam}.jpg"
                ps = args.out / args.shift_kind / f"shift_{cam}.jpg"
                hc = save_jpg(arr_c, pc); hs = save_jpg(arr_s, ps)
                images[cam] = {"clean": {"path": str(pc), "sha256": hc},
                               "shift": {"path": str(ps), "sha256": hs}}
            except Exception as exc:
                images[cam] = {"error": repr(exc)}
        rec = {
            "clean": {"suite": args.clean_suite, "task": args.clean_task, "bddl": str(clean_bddl)},
            "shift": {"suite": args.shift_suite, "task": args.shift_task, "bddl": str(shift_bddl),
                      "kind": args.shift_kind},
            "init_index": args.init_index,
            "state_digest_clean": digest(snap_c),
            "state_digest_restored": digest(snap_s),
            "compare": cmp,
            "camera_equal": bool(cam_eq),
            "camera_expected_equal": bool(expected_cam_equal),
            "camera_ok": bool(cam_eq == expected_cam_equal),
            "body_names_equal": cmp["pass"]["body_names"],
            "images": images,
            "qpos_sha256": sha(np.asarray(snap_c["qpos"])),
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "pair_manifest.jsonl").write_text(json.dumps(rec, indent=2))
        (args.out / "pair_manifest.json").write_text(json.dumps(rec, indent=2))
        print(json.dumps({k: rec[k] for k in ("compare", "camera_equal", "camera_ok", "body_names_equal")}, indent=2))
        if not cmp["all_pass"]:
            print("WARNING: numerical state equality failed; inspect errors.", file=sys.stderr)
        if not rec["camera_ok"]:
            print("WARNING: camera equality differs from expectation.", file=sys.stderr)
    finally:
        env_c.close(); env_s.close()


if __name__ == "__main__":
    main()
