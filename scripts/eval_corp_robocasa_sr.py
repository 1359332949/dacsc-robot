"""CoRP-RoboCasa success rate + Gaussian blur shift (RoboCasa-Shift 风格)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

ROOT = Path("/root/dacsc-robot")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "robocasa"))

from dacsc.corp_robocasa import load_adapted_checkpoint  # noqa: E402
from dacsc.corp_data import LinearNormalizer  # noqa: E402
from dacsc.corp_robocasa import ROBOCASA_CORP_CONFIG  # noqa: E402
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv  # noqa: E402
from robocasa.utils.env_utils import convert_action  # noqa: E402

VIEWS = ["robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"]
OBS_KEY = {
    "robot0_agentview_left": "video.robot0_agentview_left",
    "robot0_eye_in_hand": "video.robot0_eye_in_hand",
    "robot0_agentview_right": "video.robot0_agentview_right",
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def policy_inputs(obs, lang, norm, device, blur_sigma=0.0):
    images = {}
    for v in VIEWS:
        img = np.asarray(obs[OBS_KEY[v]], dtype=np.uint8)  # HWC
        pil = Image.fromarray(img)
        if blur_sigma > 0:
            pil = pil.filter(ImageFilter.GaussianBlur(radius=float(blur_sigma)))
        pil = pil.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        images[v] = ((x - MEAN) / STD).unsqueeze(0).to(device)
    state = torch.cat([
        torch.as_tensor(obs["state.end_effector_position_relative"], dtype=torch.float32),
        torch.as_tensor(obs["state.end_effector_rotation_relative"], dtype=torch.float32),
        torch.as_tensor(obs["state.gripper_qpos"], dtype=torch.float32),
    ]).unsqueeze(0).to(device)
    state = norm.normalize_state(state)
    return images, state, [lang]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="CloseFridge")
    ap.add_argument("--styles", default="11")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--action-steps", type=int, default=8)
    ap.add_argument("--blur-sigma", type=float, default=0.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="/hy-tmp/runs/corp_robocasa_sr/sr.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, (m, s) = load_adapted_checkpoint(args.model, device=device)
    norm = LinearNormalizer(m, s)
    model.eval()
    styles = [int(x) for x in args.styles.split(",")]

    env = RoboCasaGymEnv(env_name=args.task, split=None, obj_instance_split="pretrain",
                         style_ids=styles, camera_widths=256, camera_heights=256)
    records = []
    started = time.monotonic()
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=ep)
            lang = env.env.get_ep_meta().get("lang", args.task)
            success, steps = False, 0
            while not success and steps < args.max_steps:
                images, state, instr = policy_inputs(obs, lang, norm, device, args.blur_sigma)
                with torch.no_grad():
                    chunk = model.sample_action(images, state, instr).float().cpu().numpy()[0]
                for a in chunk[:args.action_steps]:
                    obs, reward, done, trunc, info = env.step(convert_action(np.clip(a, -1, 1).astype(np.float32)))
                    steps += 1
                    success = bool(info.get("success", False))
                    if success or done or steps >= args.max_steps:
                        break
            records.append({"episode": ep, "style_ids": styles, "blur_sigma": args.blur_sigma,
                            "success": success, "steps": steps})
            print(f"[{args.task}] ep={ep} style={styles} blur={args.blur_sigma} "
                  f"success={int(success)} steps={steps}", flush=True)
    finally:
        env.close()

    sr = sum(r["success"] for r in records) / len(records)
    out = {"task": args.task, "styles": styles, "blur_sigma": args.blur_sigma,
           "episodes": len(records), "successes": sum(r["success"] for r in records),
           "success_rate": sr, "elapsed_seconds": time.monotonic() - started}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": out, "records": records}, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
