"""Launch a phase-3 frozen/pilot (or formal) batch from configs/phase3_tasks.json."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frozen", "salt", "sg_salt"], default="frozen")
    ap.add_argument("--suite-filter", default=None)
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--tasks-file", default="configs/phase3_pilot_tasks.json")
    ap.add_argument("--out-root", default="/hy-tmp/validation/p3_pilot2")
    ap.add_argument("--lambda-protect", type=float, default=5.0)
    ap.add_argument("--a-basis", default="/hy-tmp/validation/interface_stats_spatial_k32_big2.json")
    ap.add_argument("--calibration", default="/hy-tmp/validation/p11_calib/gate.json")
    ap.add_argument("--extra", default="")
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.tasks_file).read_text())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    })
    launched = []
    for cell, info in cfg["cells"].items():
        if args.suite_filter and info["suite"] != args.suite_filter:
            continue
        name = cell.replace("|", "_")
        tasks = ",".join(map(str, info.get("tasks", info.get("candidates", []))))
        out = out_root / f"{name}.jsonl"
        cmd = [str(ROOT / "repos/CoRP/.venv/bin/python"), str(ROOT / "scripts/salt_eval.py"),
               "--suite", info["suite"], "--tasks", tasks,
               "--episodes-per-task", str(args.episodes), "--seed", str(args.seed),
               "--mode", args.mode, "--exec-len", "6",
               "--libero-src", "/hy-tmp/libero_plus",
               "--libero-config-path", "/hy-tmp/libero_plus_config",
               "--out", str(out)]
        if args.mode == "sg_salt":
            cmd += ["--lambda-protect", str(args.lambda_protect), "--a-basis", args.a_basis]
        if args.mode != "frozen":
            cmd += ["--calibration", args.calibration]
        if args.extra:
            cmd += args.extra.split()
        log = open(out_root / f"{name}.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env, cwd=str(ROOT))
        launched.append((name, proc.pid))
        print("launched", name, "pid", proc.pid, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
