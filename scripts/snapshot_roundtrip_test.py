"""Fail-closed MuJoCo snapshot/restore round-trip + short evolution check."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from dacsc.state_pairing import compare, digest, model_compatible, restore, snapshot  # noqa: E402
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS  # noqa: E402

EXTRA = ("ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "eq_active")


def configure(args):
    import os, yaml
    src = args.libero_src.expanduser().resolve(); root = src / "libero" / "libero"
    sys.path.insert(0, str(src))
    cfg = args.libero_config.expanduser().resolve(); cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "datasets").mkdir(exist_ok=True)
    cfg_path = cfg / "config.yaml"
    if not cfg_path.is_file():
        cfg_path.write_text(yaml.safe_dump({
            "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
            "init_states": str(root / "init_files"), "assets": str(root / "assets"),
            "datasets": str(cfg / "datasets")}))
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-bddl", type=Path, required=True)
    ap.add_argument("--clean-init-file", type=Path, required=True)
    ap.add_argument("--state-index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    init = __import__("torch").load(args.clean_init_file, weights_only=False)
    env_a = OffScreenRenderEnv(bddl_file_name=str(args.clean_bddl), camera_heights=256,
                               camera_widths=256, render_gpu_device_id=0)
    env_b = OffScreenRenderEnv(bddl_file_name=str(args.clean_bddl), camera_heights=256,
                               camera_widths=256, render_gpu_device_id=0)
    out = {"state_index": args.state_index, "extra_fields": {}}
    try:
        env_a.seed(0); env_a.reset(); env_a.set_init_state(init[args.state_index])
        for _ in range(SETTLING_STEPS):
            env_a.step(DUMMY_ACTION)
        sa = snapshot(env_a)
        env_b.seed(0); env_b.reset(); restore(env_b, sa)
        sb = snapshot(env_b)
        out["model_compatible"] = model_compatible(sa, sb, strict_bodies=True,
                                                   strict_geometry=True, strict_physics=True)
        cmp = compare(sa, sb); out["state_compare"] = cmp
        for k in EXTRA:
            a = np.asarray(sa.get(k, []), dtype=float); b = np.asarray(sb.get(k, []), dtype=float)
            if a.shape != b.shape:
                out["extra_fields"][k] = {"pass": False, "shape": [list(a.shape), list(b.shape)]}
            elif a.size == 0:
                out["extra_fields"][k] = {"pass": True, "empty": True}
            else:
                err = float(np.max(np.abs(a - b)))
                out["extra_fields"][k] = {"pass": bool(np.isfinite(err) and err <= 1e-6), "max_abs": err}
        # Direct physics evolution with identical ctrl.  High-level env.step()
        # also advances the OSC controller's internal state, which is outside
        # the MuJoCo snapshot contract and not needed for occupancy rendering.
        ctrl = np.asarray(sa.get("ctrl", []), dtype=float)
        if ctrl.size:
            env_a.sim.data.ctrl[:] = ctrl; env_b.sim.data.ctrl[:] = ctrl
        for _ in range(args.steps):
            env_a.sim.step(); env_b.sim.step()
        ea = snapshot(env_a); eb = snapshot(env_b)
        ev = compare(ea, eb); out["evolution_compare"] = ev
        out["digest_match"] = digest(sa) == digest(sb)
        out["all_pass"] = bool(out["model_compatible"]["pass"] and cmp["all_pass"]
                               and all(v["pass"] for v in out["extra_fields"].values())
                               and ev["all_pass"])
    finally:
        env_a.close(); env_b.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"all_pass": out["all_pass"], "state": out["state_compare"]["all_pass"],
                      "model": out["model_compatible"]["pass"],
                      "extra": {k: v["pass"] for k, v in out["extra_fields"].items()},
                      "evolution": out["evolution_compare"]["all_pass"]}, indent=2))
    if not out["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
