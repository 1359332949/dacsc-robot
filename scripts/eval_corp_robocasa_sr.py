"""CoRP-RoboCasa success rate with the declared observation-side image protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "robocasa"))

from dacsc.corp_robocasa import load_adapted_checkpoint  # noqa: E402
from dacsc.corp_data import LinearNormalizer  # noqa: E402
from dacsc.corp_images import prepare_image, image_contract
from dacsc.rollout_protocol import prepare_episode, rollout_episode  # noqa: E402
from dacsc.eval_protocol import PROTOCOL_VERSION, file_sha256  # noqa: E402
from dacsc.run_output import entry_file  # noqa: E402
from dacsc.upstream import verify_upstream

VIEWS = ["robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"]
OBS_KEY = {
    "robot0_agentview_left": "video.robot0_agentview_left",
    "robot0_eye_in_hand": "video.robot0_eye_in_hand",
    "robot0_agentview_right": "video.robot0_agentview_right",
}


def parse_styles(value: str) -> list[int]:
    styles = []
    for part in value.split(","):
        bounds = part.strip().split("-")
        if len(bounds) == 1:
            lo = hi = int(bounds[0])
        elif len(bounds) == 2:
            lo, hi = map(int, bounds)
        else:
            raise ValueError("styles must be integers or ascending ranges")
        if not 11 <= lo <= hi <= 60:
            raise ValueError(
                "styles must be ascending native pretrain IDs within 11..60"
            )
        styles.extend(range(lo, hi + 1))
    if len(styles) != len(set(styles)):
        raise ValueError("styles must be unique")
    return styles


def policy_inputs(obs, lang, norm, device, blur_sigma=0.0, shift="none"):
    images = {}
    for v in VIEWS:
        img = np.asarray(obs[OBS_KEY[v]], dtype=np.uint8)  # HWC
        images[v] = (
            prepare_image(img, shift, blur_sigma=blur_sigma).unsqueeze(0).to(device)
        )
    state = (
        torch.cat(
            [
                torch.as_tensor(
                    obs["state.end_effector_position_relative"], dtype=torch.float32
                ),
                torch.as_tensor(
                    obs["state.end_effector_rotation_relative"], dtype=torch.float32
                ),
                torch.as_tensor(obs["state.gripper_qpos"], dtype=torch.float32),
            ]
        )
        .unsqueeze(0)
        .to(device)
    )
    state = norm.normalize_state(state)
    return images, state, [lang]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="CloseFridge")
    ap.add_argument("--styles", type=parse_styles, default="11")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--action-steps", type=int, default=8)
    ap.add_argument("--blur-sigma", type=float, default=0.0)
    ap.add_argument(
        "--shift",
        default="none",
        choices=["none", "grayscale", "color_jitter", "low_contrast", "hue_shift"],
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layout-id", type=int, default=11)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--manifest-mode", choices=["create", "replay"], default="replay")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.episodes <= 0 or args.action_steps <= 0:
        ap.error("episodes and action-steps must be positive")
    with entry_file(Path(args.out)):
        return run(args)


def run(args) -> int:
    upstream = verify_upstream()
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv
    from robocasa.utils.env_utils import convert_action
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.models.scenes.scene_registry import LayoutType, unpack_layout_ids
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    # Match official DP evaluation: per-task horizon, source/pretrain layouts.
    layouts = unpack_layout_ids(LayoutType.TRAIN)
    max_steps = int(get_task_horizon(args.task))
    if max_steps <= 0:
        raise ValueError("official task horizon must be positive")
    if args.layout_id not in layouts:
        raise ValueError("layout-id must be in the official pretrain layout group")
    checkpoint_digest = file_sha256(Path(args.model))

    device = torch.device(args.device)
    model, (m, s) = load_adapted_checkpoint(args.model, device=device)
    norm = LinearNormalizer(m, s)
    model.eval()
    if args.action_steps > model.action_horizon:
        raise ValueError("action-steps exceeds the checkpoint action horizon")
    styles = args.styles
    style_order = [styles[ep % len(styles)] for ep in range(args.episodes)]

    protocol = {
        "version": PROTOCOL_VERSION,
        "checkpoint_sha256": checkpoint_digest,
        "shift": args.shift,
        "blur_sigma": args.blur_sigma,
        "layouts": layouts,
        "layout_id": args.layout_id,
        "base_seed": args.seed,
        "action_steps": args.action_steps,
        "action_horizon": model.action_horizon,
        "euler_steps": model.config["action_generator"]["num_sample_steps"],
        "image_size": model.config["image_size"],
        "precision": "fp32",
        "horizon_source": "robocasa.utils.dataset_registry_utils.get_task_horizon",
        "images": image_contract(),
        "upstream": upstream,
        "training": model.evaluation_origin,
    }

    env = RoboCasaGymEnv(
        env_name=args.task,
        split=None,
        obj_instance_split="pretrain",
        style_ids=styles,
        layout_ids=LayoutType.TRAIN,
        camera_widths=256,
        camera_heights=256,
        seed=args.seed,
    )
    records = []
    started = time.monotonic()
    try:
        for ep in range(args.episodes):
            obs, manifest, digest = prepare_episode(
                env,
                args.manifest_dir / f"episode_{ep:06d}.json.gz",
                task=args.task,
                styles=styles,
                episode=ep,
                seed=args.seed + ep,
                layouts=layouts,
                mode=args.manifest_mode,
                reset_to=reset_to,
                requested_style=style_order[ep],
                requested_layout=args.layout_id,
            )
            result = rollout_episode(
                env,
                model,
                norm,
                device,
                obs,
                manifest["ep_meta"]["lang"],
                policy_inputs=policy_inputs,
                convert_action=convert_action,
                max_steps=max_steps,
                action_steps=args.action_steps,
                blur_sigma=args.blur_sigma,
                shift=args.shift,
            )
            records.append(
                {
                    "episode": ep,
                    "style_ids": styles,
                    "actual_style": manifest["actual_style"],
                    "actual_layout": manifest["actual_layout"],
                    "requested_style": manifest["requested_style"],
                    "requested_layout": manifest["requested_layout"],
                    "environment_seed": manifest["environment_seed"],
                    "policy_seed": manifest["policy_seed"],
                    "manifest_sha256": digest,
                    "blur_sigma": args.blur_sigma,
                    "shift": args.shift,
                    **result,
                }
            )
            print(
                f"[{args.task}] ep={ep} style={styles} blur={args.blur_sigma} "
                f"success={int(result['success'])} steps={result['steps']}",
                flush=True,
            )
    finally:
        env.close()

    sr = sum(r["success"] for r in records) / len(records)
    out = {
        "task": args.task,
        "styles": styles,
        "blur_sigma": args.blur_sigma,
        "shift": args.shift,
        "model": str(Path(args.model).resolve()),
        "seed": args.seed,
        "manifest_dir": str(args.manifest_dir.resolve()),
        "max_steps": max_steps,
        "episodes": len(records),
        "layout_id": args.layout_id,
        "style_order": style_order,
        "successes": sum(r["success"] for r in records),
        "success_rate": sr,
        "elapsed_seconds": time.monotonic() - started,
    }
    if file_sha256(Path(args.model)) != checkpoint_digest:
        raise RuntimeError("checkpoint changed during evaluation")
    Path(args.out).write_text(
        json.dumps({"protocol": protocol, "summary": out, "records": records}, indent=2)
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
