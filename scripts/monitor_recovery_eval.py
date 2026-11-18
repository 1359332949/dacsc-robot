"""Closed-loop monitor + optional chunk-interrupt/reflex evaluation.

Requires a trained CausalFailureMonitor checkpoint. This script is the harness
for the future experiment; it is not run without a validated monitor.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from target_only_action_validation import configure, visual_from_obs  # noqa: E402
from dacsc.failure_monitor import CausalFailureMonitor  # noqa: E402
from dacsc.monitor_ensemble import MonitorEnsemble  # noqa: E402
from dacsc.tta_protocol import load_split, resolve_indices, seeds_for_role, sha256_file, validate_split_identity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--role", default="target_dev")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--domain", choices=["shift", "clean"], default="shift")
    ap.add_argument("--lighting-id", type=int, default=2292)

    ap.add_argument("--modes", default="frozen,shadow")
    ap.add_argument("--allow-intervention", action="store_true",
                    help="required for mode=monitor; default is shadow only")

    ap.add_argument("--reflex", action="store_true",
                    help="deprecated unsafe blind retreat; requires explicit opt-in")
    ap.add_argument("--interrupt-k", type=int, default=4,
                    help="temporary executed_k after an alarm")
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--max-interventions", type=int, default=2)
    ap.add_argument("--retreat-z", type=float, default=0.35)
    ap.add_argument("--reflex-steps", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    if "monitor" in args.modes.split(",") and not args.allow_intervention:
        raise ValueError("mode=monitor requires --allow-intervention; use shadow for passive runs")
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == args.suite)
    if args.role != "target_dev":
        raise ValueError("monitor recovery is restricted to the frozen target_dev role")
    split = load_split(args.split_json)
    validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    artifact = json.loads(args.artifact.read_text())
    if artifact.get("role") != args.role:
        raise ValueError("role does not match monitor artifact")
    if artifact.get("split_sha256") != sha256_file(args.split_json):
        raise ValueError("split hash does not match monitor artifact")
    if artifact.get("schema") != "monitor_artifact_v2":
        raise ValueError("recovery requires monitor_artifact_v2 ensemble")
    if int(artifact.get("min_members", 0)) < 2:
        raise ValueError("ensemble must contain at least two independent members")
    ensemble = MonitorEnsemble(args.artifact)
    threshold = float(artifact["threshold"])
    persistence = int(artifact["persistence"])
    stride = int(artifact["stride"])
    window = int(artifact["window"])
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == args.lighting_id)
    env_bddl = variant["base_bddl"] if args.domain == "clean" else variant["env_bddl_name"]
    ids = resolve_indices(split, args.role, "")
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    W = window
    seeds = seeds_for_role(split, args.role)
    policy = load_policy(args.checkpoint, device=torch.device(args.device)); policy.eval()
    records = []
    for state_id in ids:
        for seed in seeds:
            for mode in args.modes.split(","):
                env = OffScreenRenderEnv(bddl_file_name=env_bddl, camera_heights=256,
                                         camera_widths=256, render_gpu_device_id=args.render_gpu)
                try:
                    np.random.seed(seed); torch.manual_seed(seed)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                    env.seed(seed); env.reset(); obs = env.set_init_state(states[state_id])
                    for _ in range(SETTLING_STEPS): obs, _, _, _ = env.step(DUMMY_ACTION)
                    steps = 0; success = False; done = False; interventions = 0; alarms = 0
                    current_k = args.executed_k
                    alarm_run = 0
                    buf_s, buf_a, buf_v = [], [], []
                    prev_vis_raw = None
                    while not success and not done and steps < args.max_steps:
                        chunk_alarm = False
                        po = policy_observation(obs, language, policy)
                        with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                            enabled=args.device.startswith("cuda")):
                            controller = policy.predict(po)[0].float().cpu().numpy()
                        for k in range(current_k):
                            raw = controller[k]; clipped = np.clip(raw, -1.0, 1.0)
                            obs, reward, done, info = env.step(clipped); steps += 1
                            success = is_success(env, info, reward, done)
                            po_after = policy_observation(obs, language, policy)
                            st = po_after["state"][0].detach().cpu().numpy().astype(float)
                            vis, _ = visual_from_obs(policy, po_after); vis = vis.mean(dim=1)[0].detach().float().cpu().numpy().astype(float)
                            delta = np.zeros_like(vis) if prev_vis_raw is None else (vis - prev_vis_raw)
                            prev_vis_raw = vis
                            buf_s.append(st); buf_a.append(clipped); buf_v.append(delta)
                            if success or done or steps >= args.max_steps: break
                            if (mode in ("monitor", "shadow") and len(buf_s) >= W and steps % stride == 0
                                    and (mode == "shadow" or interventions < args.max_interventions)):
                                prob = float(ensemble.predict(
                                    np.asarray(buf_s[-W:], dtype=np.float32)[None, ...],
                                    np.asarray(buf_a[-W:], dtype=np.float32)[None, ...],
                                    np.asarray(buf_v[-W:], dtype=np.float32)[None, ...])[0])
                                alarm_run = alarm_run + 1 if prob >= threshold else 0
                                if alarm_run >= persistence:
                                    alarms += 1
                                    if mode == "shadow":
                                        alarm_run = 0
                                        continue
                                    interventions += 1
                                    chunk_alarm = True
                                    if args.reflex:
                                        raise RuntimeError("blind reflex is disabled by design; use elastic interruption")
                                    current_k = max(1, min(args.interrupt_k, args.executed_k))
                                    buf_s, buf_a, buf_v = [], [], []
                                    prev_vis_raw = None
                                    alarm_run = 0
                                    break
                    # Keep the shortened horizon for the next chunk after an
                    # alarm; restore it only after a subsequent quiet chunk.
                    if current_k < args.executed_k and not chunk_alarm:
                        current_k = args.executed_k
                    rec = {"state_id": state_id, "seed": seed, "mode": mode, "success": bool(success),
                           "steps": steps, "alarms": alarms, "interventions": interventions,
                           "persistence": persistence, "monitor_stride": stride,
                           "artifact_sha256": sha256_file(args.artifact),
                           "ensemble_members": len(ensemble.members),
                           "hit_max_steps": bool(steps >= args.max_steps and not success)}
                    records.append(rec); print(json.dumps(rec), flush=True)
                finally:
                    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "monitor_recovery_eval_v1", "records": records}, indent=2) + "\n")
    for mode in args.modes.split(","):
        rs = [r for r in records if r["mode"] == mode]
        print(mode, sum(r["success"] for r in rs), "/", len(rs))


if __name__ == "__main__":
    main()
