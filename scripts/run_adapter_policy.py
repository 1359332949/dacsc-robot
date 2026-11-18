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
    monitor = MonitorEnsemble(args.monitor_artifact) if "gated" in ("frozen", "adapter", "gated") else None
    mthr = float(monitor.artifact["threshold"]); mpers = int(monitor.artifact["persistence"])
    mstride = int(monitor.artifact["stride"]); mwin = int(monitor.artifact["window"])
    records = []
    seeds = ([int(x) for x in args.seeds.split(",")] if args.seeds else [args.seed])
    for state_id in ids:
        for seed in seeds:
            for mode in args.modes.split(","):
                env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                         render_gpu_device_id=args.render_gpu)
                try:
                    np.random.seed(seed); torch.manual_seed(seed)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                    env.seed(seed); env.reset(); obs = env.set_init_state(states[state_id])
                    for _ in range(SETTLING_STEPS): obs, _, _, _ = env.step(DUMMY_ACTION)
                    steps = 0; success = False; done = False; chunk = 0
                    gate_active = False; alarm_run = 0
                    monitor_probs = []; monitor_alarms = 0; gate_step = None
                    buf_s, buf_a, buf_v = [], [], []
                    prev_vis = None
                    while not success and not done and steps < args.max_steps:
                        po = policy_observation(obs, language, policy)
                        torch.manual_seed(seed + 100003 * state_id + chunk)
                        z0 = torch.randn((1, policy.action_generator.action_horizon,
                                          policy.action_generator.action_dim), device=device, dtype=torch.float32)
                        if mode == "frozen":
                            out = forward_with_adapter(policy, po, zero, z0); used = False
                        elif mode == "adapter":
                            out = forward_with_adapter(policy, po, adapter, z0); used = True
                        elif mode == "first":
                            used = chunk == 0
                            out = forward_with_adapter(policy, po, adapter if used else zero, z0)
                        else:
                            used = bool(gate_active)
                            out = forward_with_adapter(policy, po, adapter if used else zero, z0)
                        actions = out["controller_actions"][0].detach().float().cpu().numpy()
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
                    records.append({"state_id": state_id, "seed": seed, "mode": mode, "success": bool(success),
                                    "gate_used": bool(used), "gate_active_final": bool(gate_active),
                                    "gate_step": gate_step, "monitor_alarm_windows": monitor_alarms,
                                    "monitor_max_prob": max(monitor_probs) if monitor_probs else None,
                                    "monitor_n_evals": len(monitor_probs),
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
