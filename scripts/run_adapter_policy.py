"""Closed-loop evaluation of a trained CoRP visual action adapter."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP")); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, is_success, policy_observation  # noqa: E402
from target_only_action_validation import visual_from_obs  # noqa: E402
from corp import load_policy  # noqa: E402
from target_only_action_validation import configure  # noqa: E402
from dacsc.action_adapter import ResidualVisualAdapter  # noqa: E402
from dacsc.monitor_ensemble import MonitorEnsemble  # noqa: E402
from dacsc.state_pairing import snapshot, digest, digest_model, digest_controller  # noqa: E402
from dacsc.differentiable_policy import forward_with_adapter  # noqa: E402
from dacsc.tta_protocol import load_split, resolve_indices, seeds_for_role, validate_split_identity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--domain", choices=["shift", "clean"], default="shift")
    ap.add_argument("--state-ids", default="")
    ap.add_argument("--seed", type=int, default=7777)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--env-seeds", default="")
    ap.add_argument("--z0-seeds", default="")
    ap.add_argument("--monitor-artifact", type=Path, default=Path("docs/data/monitor_artifact_v2.json"))
    ap.add_argument("--modes", default="frozen,adapter,gated,first")
    ap.add_argument("--gate-persistence", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    import os
    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise RuntimeError("adapter evaluation requires MUJOCO_GL=osmesa")
    if args.out.exists(): raise FileExistsError(args.out)
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == args.suite)
    split = load_split(args.split_json); validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == args.lighting_id)
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    ids = resolve_indices(split, "target_dev", args.state_ids)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    bddl = variant["env_bddl_name"] if args.domain == "shift" else variant["base_bddl"]
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    for p in policy.parameters(): p.requires_grad_(False)
    ck0 = torch.load(args.adapter, map_location="cpu")
    adapter = ResidualVisualAdapter(768, ck0["rank"]).to(device); adapter.load_state_dict(ck0["adapter"]); adapter.eval()
    zero = ResidualVisualAdapter(768, ck0["rank"]).to(device); zero.eval()
    monitor = MonitorEnsemble(args.monitor_artifact) if "gated" in args.modes.split(",") else None
    if monitor is not None:
        mthr = float(monitor.artifact["threshold"]); mpers = int(monitor.artifact["persistence"])
        mstride = int(monitor.artifact["stride"]); mwin = int(monitor.artifact["window"])
    records = []
    legacy_seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed])
    env_seeds = ([int(x) for x in args.env_seeds.split(",")] if args.env_seeds else legacy_seeds)
    z0_seeds = ([int(x) for x in args.z0_seeds.split(",")] if args.z0_seeds else legacy_seeds)
    adapter_sha = __import__("hashlib").sha256(args.adapter.read_bytes()).hexdigest()
    for state_id in ids:
        for env_seed in env_seeds:
          for z0_seed in z0_seeds:
            for mode in args.modes.split(","):
                env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                         render_gpu_device_id=args.render_gpu)
                try:
                    np.random.seed(env_seed); torch.manual_seed(env_seed)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(env_seed)
                    env.seed(env_seed); env.reset(); obs = env.set_init_state(states[state_id])
                    for _ in range(SETTLING_STEPS): obs, _, _, _ = env.step(DUMMY_ACTION)
                    init_snap = snapshot(env)
                    init_state_digest = digest(init_snap)
                    init_model_digest = digest_model(init_snap)
                    init_controller_digest = digest_controller(init_snap)
                    init_obs_hash = __import__("hashlib").sha256(
                        np.asarray(obs.get("agentview_image", []), dtype=np.uint8).tobytes()).hexdigest()
                    steps = 0; success = False; done = False; chunk = 0
                    adapter_forward_calls = 0; adapter_applied_chunks = 0
                    adapter_first_applied_step = None; adapter_last_applied_step = None
                    first_z0_hash = None; z0_hashes = []; chunk_log = []
                    gate_active = False; alarm_run = 0
                    monitor_probs = []; monitor_alarms = 0; gate_step = None
                    buf_s, buf_a, buf_v = [], [], []
                    prev_vis = None
                    while not success and not done and steps < args.max_steps:
                        po = policy_observation(obs, language, policy)
                        torch.manual_seed(z0_seed + 100003 * state_id + chunk)
                        z0 = torch.randn((1, policy.action_generator.action_horizon,
                                          policy.action_generator.action_dim), device=device, dtype=torch.float32)
                        z0_hash = __import__("hashlib").sha256(z0.detach().float().cpu().numpy().tobytes()).hexdigest()
                        z0_hashes.append(z0_hash)
                        if first_z0_hash is None: first_z0_hash = z0_hash
                        frozen_out = forward_with_adapter(policy, po, zero, z0)
                        adapter_out = forward_with_adapter(policy, po, adapter, z0)
                        if mode == "frozen":
                            out = frozen_out; used = False
                        elif mode == "adapter":
                            out = adapter_out; used = True
                            adapter_forward_calls += 1
                            if adapter_first_applied_step is None: adapter_first_applied_step = steps
                            adapter_last_applied_step = steps
                        elif mode == "first":
                            used = chunk == 0
                            out = adapter_out if used else frozen_out
                            if used:
                                adapter_forward_calls += 1
                                if adapter_first_applied_step is None: adapter_first_applied_step = steps
                                adapter_last_applied_step = steps
                        else:
                            used = bool(gate_active)
                            if used and adapter_first_applied_step is None: adapter_first_applied_step = steps
                            if used: adapter_last_applied_step = steps
                            out = adapter_out if used else frozen_out
                            if used: adapter_forward_calls += 1
                        if used: adapter_applied_chunks += 1
                        actions = out["controller_actions"][0].detach().float().cpu().numpy()
                        f_ctrl = frozen_out["controller_actions"][0].detach().float().cpu().numpy()
                        a_ctrl = adapter_out["controller_actions"][0].detach().float().cpu().numpy()
                        f_norm = frozen_out["normalized_actions"][0].detach().float().cpu().numpy()
                        a_norm = adapter_out["normalized_actions"][0].detach().float().cpu().numpy()
                        chosen_clip = np.clip(actions, -1.0, 1.0)
                        clip_ratio = float(np.mean(np.abs(actions) > 1.0))
                        chunk_log.append({
                            "chunk": chunk, "steps_start": steps, "mode": mode, "adapter_used": bool(used),
                            "z0_seed": z0_seed, "z0_sha256": z0_hash,
                            "norm_delta_mae": float(np.mean(np.abs(a_norm - f_norm))),
                            "controller_delta_rmse": float(np.sqrt(np.mean((a_ctrl - f_ctrl) ** 2))),
                            "clipped_delta_rmse": float(np.sqrt(np.mean((chosen_clip - np.clip(f_ctrl, -1, 1)) ** 2))),
                            "clip_ratio": clip_ratio,
                            "clip_ratio_xyz": float(np.mean(np.abs(actions[:, :3]) > 1.0)),
                            "clip_ratio_grip": float(np.mean(np.abs(actions[:, 6]) > 1.0)),
                        })
                        for k in range(min(args.executed_k, len(actions))):
                            obs, reward, done, info = env.step(np.clip(actions[k], -1.0, 1.0)); steps += 1
                            success = is_success(env, info, reward, done)
                            if mode == "gated":
                                po_after = policy_observation(obs, language, policy)
                                st = po_after["state"][0].detach().float().cpu().numpy()
                                vis, _ = visual_from_obs(policy, po_after)
                                vis = vis.mean(dim=1)[0].detach().float().cpu().numpy()
                                delta = np.zeros_like(vis) if prev_vis is None else vis - prev_vis
                                prev_vis = vis
                                buf_s.append(st); buf_a.append(np.clip(actions[k], -1.0, 1.0)); buf_v.append(delta)
                                if len(buf_s) >= mwin and steps % mstride == 0:
                                    prob = float(monitor.predict(
                                        np.asarray(buf_s[-mwin:], dtype=np.float32)[None, ...],
                                        np.asarray(buf_a[-mwin:], dtype=np.float32)[None, ...],
                                        np.asarray(buf_v[-mwin:], dtype=np.float32)[None, ...])[0])
                                    monitor_probs.append(prob)
                                    if prob >= mthr: monitor_alarms += 1
                                    alarm_run = alarm_run + 1 if prob >= mthr else 0
                                    if alarm_run >= args.gate_persistence and not gate_active:
                                        gate_active = True; gate_step = steps
                            if success or done or steps >= args.max_steps: break
                        chunk += 1
                    records.append({"state_id": state_id, "env_seed": env_seed, "z0_seed": z0_seed,
                                    "mode": mode, "success": bool(success),
                                    "adapter_enabled": bool(mode in ("adapter", "first", "gated") and used),
                                    "adapter_forward_calls": adapter_forward_calls,
                                    "adapter_applied_chunks": adapter_applied_chunks,
                                    "adapter_first_applied_step": adapter_first_applied_step,
                                    "adapter_last_applied_step": adapter_last_applied_step,
                                    "adapter_checkpoint_sha256": adapter_sha,
                                    "gate_used": bool(used) if mode == "gated" else None,
                                    "gate_active_final": bool(gate_active),
                                    "gate_step": gate_step,
                                    "monitor_alarm_windows": monitor_alarms,
                                    "monitor_max_prob": max(monitor_probs) if monitor_probs else None,
                                    "monitor_n_evals": len(monitor_probs),
                                    "first_z0_sha256": first_z0_hash,
                                    "z0_sha256s": z0_hashes,
                                    "chunk_log": chunk_log,
                                    "init_state_digest": init_state_digest,
                                    "init_model_digest": init_model_digest,
                                    "init_controller_digest": init_controller_digest,
                                    "init_obs_sha256": init_obs_hash,
                                    "steps": steps, "hit_max_steps": bool(steps >= args.max_steps and not success)})
                    print(json.dumps(records[-1]), flush=True)
                finally:
                    env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema": "adapter_policy_eval_v1", "records": records}, indent=2) + "\n")
    for mode in args.modes.split(","):
        rs = [r for r in records if r["mode"] == mode]
        if rs:
            print(mode, sum(r["success"] for r in rs), "/", len(rs))


if __name__ == "__main__":
    main()
