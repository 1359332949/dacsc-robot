"""Build identifiability-weighted PSC statistics for the released CoRP policy.

Reference implementation: SSA/PSC loss and optimizer conventions follow the
official SSA code (kzkadc/regression-tta: utils/loss.py, tta/tta_engine.py);
the PSC probe bank follows arXiv:2603.09338 Eq. (11)-(16).

Our addition (single, documented): for every probe direction we measure how much
of its variability is explained by label-preserving photometric augmentations
(the "nuisance" variability). This is stored as `probe_nuisance_var` and used at
test time to down-weight alignment along directions whose observed shift cannot
be explained by nuisance variability.
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
from evaluate_libero import (  # noqa: E402
    DUMMY_ACTION,
    SETTLING_STEPS,
    configure_libero,
    policy_observation,
)
from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from dacsc.corp_images import apply_corruption  # noqa: E402
from dacsc.upstream import verify_release_checkpoint  # noqa: E402

CORRUPTIONS = ("grayscale", "low_contrast", "color_jitter", "hue_shift")


def augment(image: torch.Tensor, name: str) -> torch.Tensor:
    """image: uint8 NHWC [1,H,W,3] -> uint8 NHWC augmented (label-preserving photometric)."""
    x = image.float().permute(0, 3, 1, 2).div(255.0)
    if name == "brightness_low":
        x = TVF.adjust_brightness(x, 0.8)
    elif name == "brightness_high":
        x = TVF.adjust_brightness(x, 1.2)
    else:
        x = apply_corruption(x, name, 0.0)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=_CORP_REPO / "checkpoints/corp_libero.pt")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--tasks", default="0,1,2,3,4")
    ap.add_argument("--chunks-per-episode", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-gpu", type=int, default=0)
    ap.add_argument("--libero-src", type=Path, required=True)
    ap.add_argument("--libero-config-path", type=Path, required=True)
    ap.add_argument("--base-stats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    verify_release_checkpoint(args.checkpoint)
    configure_libero(args)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    policy = load_policy(args.checkpoint, device=args.device)
    base_stats = json.loads(args.base_stats.read_text())
    mean = torch.tensor(base_stats["mean"], dtype=torch.float32, device=args.device)
    basis = torch.tensor(base_stats["basis"], dtype=torch.float32, device=args.device)
    probes = torch.tensor(base_stats["probes"], dtype=torch.float32, device=args.device)

    def feature_of(batch: dict) -> np.ndarray:
        prepared = prepare_observation(
            batch, config=policy.config, normalizer=policy.normalizer, device=args.device
        )
        with torch.no_grad():
            rep = policy.representation_extractor(
                prepared["images"], prepared["state"], prepared["instruction"]
            )
        return rep.tokens.mean(dim=1).float().cpu().numpy()[0]

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = [int(item) for item in args.tasks.split(",")]
    task_ids = task_ids[: suite.get_num_tasks()]
    rows = []
    for task_id in task_ids:
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        initial_states = torch.load(states_path, weights_only=False)
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=args.render_gpu,
        )
        try:
            seed = args.seed + task_id * 1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.seed(seed)
            env.reset()
            observation = env.set_init_state(initial_states[0])
            for _ in range(SETTLING_STEPS):
                observation, reward, done, info = env.step(DUMMY_ACTION)
            for _ in range(args.chunks_per_episode):
                batch = policy_observation(observation, task.language, policy)
                clean = feature_of(batch)
                rows.append(clean)
                for name in CORRUPTIONS + ("brightness_low", "brightness_high"):
                    variant = dict(batch)
                    variant["images"] = {
                        key: augment(value, name) for key, value in batch["images"].items()
                    }
                    rows.append(feature_of(variant))
                with torch.no_grad():
                    chunk = policy.predict(batch)[0].float().cpu().numpy()
                for action in chunk:
                    observation, reward, done, info = env.step(np.clip(action, -1.0, 1.0))
                    if done:
                        break
                if done:
                    break
        finally:
            env.close()

    features = torch.tensor(np.stack(rows), dtype=torch.float32, device=args.device)
    u = (features - mean) @ basis.T
    p = u @ probes.T  # [N, K^2]
    nuisance_var = p.var(dim=0).cpu().numpy()
    payload = dict(base_stats)
    payload.update(
        {
            "schema": "psc_weighted_stats_v1",
            "probe_nuisance_var": nuisance_var.tolist(),
            "n_aug_frames": int(features.shape[0]),
            "augmentations": list(CORRUPTIONS) + ["brightness_low", "brightness_high"],
            "weighting": "identifiability: w_q = nu_q / (nu_q + mu_hat_q^2 + eps)",
        }
    )
    args.out.write_text(json.dumps(payload))
    print(json.dumps({"frames": int(features.shape[0]), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
