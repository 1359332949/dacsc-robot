"""E2-B smoke test: frozen clean/shift rollouts + occupancy dual rendering.

No adaptation, no training.  Captures full MuJoCo snapshots at policy-chunk
boundaries from both Clean and Shift trajectories, then restores each snapshot
to both visual domains and saves paired images.  Phase labels here are temporal
heuristics for smoke only; physical phase labels come later.
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
from sg_cpa_phase4_final import CKPT  # noqa: E402
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa: E402
from dacsc.state_pairing import digest, model_compatible, restore, snapshot  # noqa: E402
from dacsc.libero_noise import apply_agentview_noise  # noqa: E402
from dacsc.observation_render import render_policy_observation  # noqa: E402

CONDITIONS = [
    ("libero_object", "lighting", 2222),
    ("libero_object", "background", 1),
    ("libero_spatial", "camera", 735),
    ("libero_spatial", "sensor", 1481),
    ("libero_spatial", "background", 146),
]


def configure(args):
    import os, yaml
    src = args.libero_src.expanduser().resolve(); root = src / "libero" / "libero"
    sys.path.insert(0, str(src))
    cfg = args.libero_config.expanduser().resolve(); cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "datasets").mkdir(exist_ok=True)
    cp = cfg / "config.yaml"
    if not cp.is_file():
        cp.write_text(yaml.safe_dump({"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
                                      "init_states": str(root / "init_files"), "assets": str(root / "assets"),
                                      "datasets": str(cfg / "datasets")}))
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def temporal_phase(frac):
    if frac < 0.25: return "approach"
    if frac < 0.50: return "contact"
    if frac < 0.70: return "grasp_lift"
    if frac < 0.90: return "transport"
    return "align_release"


def conditioned_obs(obs, sensor_noise, seed):
    if not sensor_noise: return obs
    out, _ = apply_agentview_noise(obs, sensor_noise, seed=seed)
    return out


def run_rollout(policy, env, init_state, language, episode_seed, max_steps, sensor_noise, label):
    np.random.seed(episode_seed); torch.manual_seed(episode_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(episode_seed)
    env.seed(episode_seed)
    if label == "clean":
        env.reset()
    # Shift envs must not be reset: reset() reloads the base scene and can
    # discard Background/Lighting variant assets.
    obs = env.set_init_state(init_state)
    for _ in range(SETTLING_STEPS):
        obs, _, done, _ = env.step(DUMMY_ACTION)
        if done: break
    snaps = []; steps = 0; success = False
    while steps < max_steps:
        frac = steps / float(max_steps)
        snaps.append({"step": steps, "phase_temporal": temporal_phase(frac),
                      "success": bool(success), "snapshot": snapshot(env)})
        obs = conditioned_obs(obs, sensor_noise, episode_seed + steps)
        batch = policy_observation(obs, language, policy)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            chunk = policy.predict(batch)[0].float().cpu().numpy()
        for action in chunk:
            obs, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
            steps += 1
            success = bool(is_success(env, info, reward, done))
            if success or done or steps >= max_steps: break
        if success or done: break
    return {"label": label, "success": success, "steps": steps, "snapshots": snaps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition-index", type=int, required=True)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--checkpoint", type=Path, default=CKPT)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(); configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    suite, domain, cid = CONDITIONS[args.condition_index]
    task = next(b for b in matrix["base_tasks"] if b["suite"] == suite)
    variant = next(v for v in task["variants"][domain] if v["classification_id"] == cid)
    base_bddl = Path(variant["base_bddl"]); init_file = Path(variant["base_init"])
    language = BDDLUtils.get_problem_info(str(base_bddl))["language_instruction"]
    shift_bddl = variant["env_bddl_name"]
    sensor_noise = 0
    if domain == "sensor":
        import re
        sensor_noise = int(re.search(r"_noise_(\d+)", variant["name"]).group(1))
    init_states = torch.load(init_file, weights_only=False)
    held = task["state_split"]["heldout_indices"][:args.states]
    policy = load_policy(args.checkpoint, device=args.device); policy.eval()
    env_c = OffScreenRenderEnv(bddl_file_name=str(base_bddl), camera_heights=256,
                               camera_widths=256, render_gpu_device_id=0)
    env_s = OffScreenRenderEnv(bddl_file_name=str(shift_bddl), camera_heights=256,
                               camera_widths=256, render_gpu_device_id=0)
    args.out.mkdir(parents=True, exist_ok=True); records = []
    try:
        for si, idx in enumerate(held):
            for label, env, noise in (("clean", env_c, 0), ("shift", env_s, sensor_noise)):
                ep_seed = 20261116 + args.condition_index * 100 + si * 10 + (0 if label == "clean" else 1)
                roll = run_rollout(policy, env, init_states[int(idx)], language, ep_seed,
                                   args.max_steps, noise, label)
                for sj, rec in enumerate(roll["snapshots"]):
                    src = rec["snapshot"]
                    pair = {}
                    for target, tenv, tnoise in (("clean", env_c, 0), ("shift", env_s, sensor_noise)):
                        compat = model_compatible(src, snapshot(tenv), strict_bodies=True,
                                                  strict_geometry=True, strict_physics=True)
                        if not compat["pass"]:
                            raise RuntimeError(f"snapshot model mismatch {suite}/{domain}/{cid} {target}")
                        obs = render_policy_observation(tenv, src, sensor_noise=tnoise,
                                                        noise_seed=1234 + int(idx))
                        raw = {"agentview": obs["agentview_image"],
                               "robot0_eye_in_hand": obs["robot0_eye_in_hand_image"]}
                        d = args.out / "images" / f"state_{idx}_{label}_{sj:02d}"
                        d.mkdir(parents=True, exist_ok=True)
                        for v, img in raw.items():
                            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(d / f"{target}_{v}.png")
                        pair[target] = {v: str(d / f"{target}_{v}.png") for v in raw}
                    diff = np.abs(np.asarray(Image.open(pair["clean"]["agentview"]), dtype=float) -
                                  np.asarray(Image.open(pair["shift"]["agentview"]), dtype=float))
                    records.append({"suite": suite, "domain": domain, "classification_id": cid,
                                    "init_index": int(idx), "episode_seed": ep_seed,
                                    "trajectory": label, "snapshot_index": sj,
                                    "step": rec["step"], "phase_temporal": rec["phase_temporal"],
                                    "success": roll["success"], "episode_steps": roll["steps"],
                                    "state_digest": digest(src), "images": pair,
                                    "agentview_mean_abs_diff": float(diff.mean())})
                print(f"{suite} {domain} {cid} state={idx} traj={label} "
                      f"success={roll['success']} steps={roll['steps']} snaps={len(roll['snapshots'])}", flush=True)
    finally:
        env_c.close(); env_s.close()
    (args.out / "smoke_metadata.json").write_text(json.dumps(records, indent=2))
    summary = {"suite": suite, "domain": domain, "classification_id": cid,
               "states": [int(x) for x in held], "records": len(records),
               "success": {label: sum(1 for r in records if r["trajectory"] == label and r["success"])
                           for label in ("clean", "shift")}}
    (args.out / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
