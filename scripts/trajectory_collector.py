"""Collect schema-complete Frozen trajectories for runtime-monitor research.

Per step: state8, pooled visual feature, raw/clipped action, reward/done,
success, chunk index. Key frames are saved every N steps. No adaptation.
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
from dacsc.tta_protocol import load_split, resolve_indices, role_ids, validate_split_identity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--role", default="target_dev")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--seeds", default="7777")
    ap.add_argument("--state-ids", default="", help="optional subset of role state ids")
    ap.add_argument("--domains", default="clean,shift")
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--executed-k", type=int, default=12)
    ap.add_argument("--frame-every", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs.bddl_utils as BDDLUtils
    matrix = json.loads(args.matrix.read_text())
    task = next(b for b in matrix["base_tasks"] if b["suite"] == args.suite)
    split = load_split(args.split_json)
    validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    variant = next(v for v in task["variants"]["lighting"] if v["classification_id"] == args.lighting_id)
    ids = resolve_indices(split, args.role, "")
    if args.state_ids.strip():
        requested = [int(x) for x in args.state_ids.split(",") if x.strip()]
        outside = set(requested) - set(ids)
        if outside:
            raise ValueError(f"state ids outside role: {sorted(outside)}")
        ids = requested
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    args.out.mkdir(parents=True, exist_ok=False)
    frames_dir = args.out / "frames"; frames_dir.mkdir()
    manifests = []
    for state_id in ids:
        for seed in [int(x) for x in args.seeds.split(",")]:
            for domain in args.domains.split(","):
                bddl = variant["base_bddl"] if domain == "clean" else variant["env_bddl_name"]
                env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                         render_gpu_device_id=args.render_gpu)
                try:
                    np.random.seed(seed); torch.manual_seed(seed)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                    env.seed(seed); env.reset(); obs = env.set_init_state(states[state_id])
                    for _ in range(SETTLING_STEPS):
                        obs, _, _, _ = env.step(DUMMY_ACTION)
                    rows = []; steps = 0; success = False; done = False; chunk = 0
                    while not success and not done and steps < args.max_steps:
                        po = policy_observation(obs, language, policy)
                        state8_pre = po["state"][0].detach().cpu().numpy().astype(float)
                        visual_pre, _ = visual_from_obs(policy, po)
                        pooled_pre = visual_pre.mean(dim=1)[0].detach().float().cpu().numpy().astype(float)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                            enabled=device.type == "cuda"):
                            controller = policy.predict(po)[0].float().cpu().numpy()
                        for k in range(args.executed_k):
                            raw = controller[k]; clipped = np.clip(raw, -1.0, 1.0)
                            if steps % args.frame_every == 0:
                                from PIL import Image
                                Image.fromarray(np.asarray(obs["agentview_image"])).save(
                                    frames_dir / f"s{state_id}_d{domain}_seed{seed}_step{steps:03d}.png")
                            obs, reward, done, info = env.step(clipped)
                            steps += 1
                            success = is_success(env, info, reward, done)
                            po_after = policy_observation(obs, language, policy)
                            state8_after = po_after["state"][0].detach().cpu().numpy().astype(float)
                            visual_after, _ = visual_from_obs(policy, po_after)
                            pooled_after = visual_after.mean(dim=1)[0].detach().float().cpu().numpy().astype(float)
                            rows.append({"step": steps, "chunk": chunk,
                                         "state8_pre": state8_pre.tolist(),
                                         "state8": state8_after.tolist(),
                                         "visual_pre": pooled_pre.tolist(),
                                         "visual_mean": pooled_after.tolist(),
                                         "raw_action": raw.tolist(),
                                         "clipped_action": clipped.tolist(), "reward": float(reward),
                                         "done": bool(done), "success": bool(success)})
                            if success or done or steps >= args.max_steps: break
                        chunk += 1
                    final_state = policy_observation(obs, language, policy)["state"][0].detach().cpu().numpy().astype(float)
                    rec = {"state_id": state_id, "seed": seed, "domain": domain,
                           "success": bool(success), "steps": steps,
                           "hit_max_steps": bool(steps >= args.max_steps and not success),
                           "final_state8": final_state.tolist(), "rows": rows}
                    (args.out / f"s{state_id}_d{domain}_seed{seed}.json").write_text(json.dumps(rec) + "\n")
                    manifests.append({k: rec[k] for k in ("state_id", "seed", "domain", "success", "steps", "hit_max_steps")})
                    print(json.dumps(manifests[-1]), flush=True)
                finally:
                    env.close()
    (args.out / "manifest.json").write_text(json.dumps({"schema": "trajectory_collector_v1",
                                                         "split_sha256": __import__("dacsc.tta_protocol", fromlist=["sha256_file"]).sha256_file(args.split_json),
                                                         "checkpoint_sha256": __import__("dacsc.tta_protocol", fromlist=["sha256_file"]).sha256_file(args.checkpoint),
                                                         "records": manifests}, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "n": len(manifests)}))


if __name__ == "__main__":
    main()
