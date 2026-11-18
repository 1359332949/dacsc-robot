"""Dump one Frozen episode trace: per-step state/action/images + montage/plot.

Diagnostic only. No adaptation and no method claim.
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
from target_only_action_validation import configure  # noqa: E402
from dacsc.tta_protocol import load_split, validate_split_identity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--state-id", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2014)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2312)
    ap.add_argument("--max-steps", type=int, default=220)
    ap.add_argument("--executed-k", type=int, default=12)
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
    states = torch.load(task["state_split"]["source_file"], weights_only=False)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device); policy.eval()
    env = OffScreenRenderEnv(bddl_file_name=variant["env_bddl_name"], camera_heights=256,
                             camera_widths=256, render_gpu_device_id=args.render_gpu)
    args.out.mkdir(parents=True, exist_ok=False)
    frames = []
    try:
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        env.seed(args.seed); env.reset(); obs = env.set_init_state(states[args.state_id])
        for _ in range(SETTLING_STEPS):
            obs, _, _, _ = env.step(DUMMY_ACTION)
        rows = []; steps = 0; success = False; done = False; chunk = 0
        while not success and not done and steps < args.max_steps:
            po = policy_observation(obs, language, policy)
            state8 = po["state"][0].detach().cpu().numpy().astype(float)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                controller = policy.predict(po)[0].float().cpu().numpy()
            for k in range(args.executed_k):
                raw = controller[k]; clipped = np.clip(raw, -1.0, 1.0)
                if steps in (0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 219):
                    frames.append((steps, np.asarray(obs["agentview_image"]).copy(),
                                   np.asarray(obs["robot0_eye_in_hand_image"]).copy()))
                obs, reward, done, info = env.step(clipped)
                steps += 1
                success = is_success(env, info, reward, done)
                rows.append({"step": steps, "chunk": chunk, "raw_action": raw.tolist(),
                             "clipped_action": clipped.tolist(), "reward": float(reward),
                             "done": bool(done), "success": bool(success),
                             "state8_before": state8.tolist()})
                if success or done or steps >= args.max_steps: break
            chunk += 1
        final_state = policy_observation(obs, language, policy)["state"][0].detach().cpu().numpy().astype(float)
        payload = {"schema": "failure_trace_v1", "state_id": args.state_id, "seed": args.seed,
                   "lighting_id": args.lighting_id, "success": bool(success), "steps": steps,
                   "hit_max_steps": bool(steps >= args.max_steps and not success),
                   "final_state8": final_state.tolist(), "rows": rows}
        (args.out / "trace.json").write_text(json.dumps(payload, indent=2) + "\n")
        # montage
        try:
            from PIL import Image
            tiles = []
            for step, agent, wrist in frames:
                a = Image.fromarray(agent); w = Image.fromarray(wrist)
                canvas = Image.new("RGB", (512, 256)); canvas.paste(a.resize((256, 256)), (0, 0)); canvas.paste(w.resize((256, 256)), (256, 0))
                tiles.append((step, canvas))
            if tiles:
                cols = min(4, len(tiles)); rows_n = (len(tiles) + cols - 1) // cols
                montage = Image.new("RGB", (512 * cols, 256 * rows_n), (255, 255, 255))
                for i, (step, im) in enumerate(tiles):
                    montage.paste(im, ((i % cols) * 512, (i // cols) * 256))
                montage.save(args.out / "frames_montage.png")
        except Exception as exc:
            (args.out / "montage_error.txt").write_text(str(exc))
        # plot if matplotlib exists
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            st = np.array([r["state8_before"] for r in rows]); act = np.array([r["clipped_action"] for r in rows])
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(st[:, 2], label="eef z"); axes[0].set_ylabel("eef z"); axes[0].legend()
            axes[1].plot(st[:, 6], label="gripper 0"); axes[1].plot(st[:, 7], label="gripper 1"); axes[1].set_ylabel("gripper"); axes[1].legend()
            axes[2].plot(np.linalg.norm(act, axis=1), label="action norm"); axes[2].set_ylabel("|a|"); axes[2].set_xlabel("step"); axes[2].legend()
            fig.tight_layout(); fig.savefig(args.out / "trace_plot.png", dpi=120)
        except Exception as exc:
            (args.out / "plot_error.txt").write_text(str(exc))
        print(json.dumps({"out": str(args.out), "success": bool(success), "steps": steps}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
