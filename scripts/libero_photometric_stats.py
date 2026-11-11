"""Unsupervised photometric-corruption statistics on LIBERO observations.

Captures observation frames with the official LIBERO env (official settling and
initial-state protocol), applies the project's photometric shifts (the same
functions used by the evaluators), and summarises simple colour statistics per
condition. Purpose: check whether the corruption FAMILY (grayscale /
low_contrast / color_jitter / hue_shift) can be identified from test-time image
statistics alone, which is required to choose the matched TTA voting axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TVF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.corp_robocasa import _CORP_REPO  # noqa: E402
sys.path.insert(0, str(_CORP_REPO / "scripts"))
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, configure_libero, policy_observation  # noqa: E402
from corp import load_policy  # noqa: E402
from dacsc.corp_images import SHIFTS, apply_corruption  # noqa: E402


def image_stats(image_uint8: torch.Tensor) -> dict:
    """image_uint8: [1,H,W,3] uint8 (post 180 rotation, as the policy sees it)."""
    rgb = image_uint8.float().permute(0, 3, 1, 2).div(255.0)  # [1,3,H,W]
    r, g, b = rgb[0, 0], rgb[0, 1], rgb[0, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    mx = rgb.max(dim=1).values
    mn = rgb.min(dim=1).values
    saturation = ((mx - mn) / mx.clamp_min(1e-6)).mean()
    hue_hist = []
    for channel in (r, g, b):
        hue_hist.append(float(channel.mean()))
    return {
        "channel_mean": [float(r.mean()), float(g.mean()), float(b.mean())],
        "luminance_mean": float(luminance.mean()),
        "luminance_std": float(luminance.std()),
        "saturation_mean": float(saturation),
        "channel_std": [float(r.std()), float(g.std()), float(b.std())],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=_CORP_REPO / "checkpoints/corp_libero.pt")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0-1")
    ap.add_argument("--frames-per-task", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, default=_CORP_REPO / "third_party" / "LIBERO")
    ap.add_argument("--libero-config-path", type=Path, default=_CORP_REPO / ".cache" / "libero")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    def parse_indices(spec: str) -> list[int]:
        out = []
        for part in spec.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                out.extend(range(int(lo), int(hi) + 1))
            else:
                out.append(int(part))
        return out

    policy = load_policy(args.checkpoint, device=args.device)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    records = []
    for task_id in parse_indices(args.tasks):
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        initial_states = torch.load(states_path, weights_only=False)
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                 render_gpu_device_id=args.render_gpu)
        try:
            env.seed(args.seed + task_id * 1000)
            env.reset()
            observation = env.set_init_state(initial_states[0])
            for _ in range(SETTLING_STEPS):
                observation, reward, done, info = env.step(DUMMY_ACTION)
            for frame in range(args.frames_per_task):
                batch = policy_observation(observation, task.language, policy)
                for shift in SHIFTS:
                    image = batch["images"]["agentview"].to(args.device)
                    if shift != "none":
                        image = apply_corruption(
                            image.float().permute(0, 3, 1, 2).div(255.0), shift, 0.0
                        ).mul(255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
                    stats = image_stats(image)
                    stats.update(task=task_id, frame=frame, shift=shift)
                    records.append(stats)
                observation, reward, done, info = env.step(
                    np.clip(policy.predict(batch)[0].float().cpu().numpy()[0], -1, 1)
                )
                if done:
                    break
        finally:
            env.close()
    payload = {"schema": "libero_photometric_stats_v1", "shifts": list(SHIFTS),
               "tasks": parse_indices(args.tasks), "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"records": len(records), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
