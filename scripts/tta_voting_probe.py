"""Test-time augmentation voting for appearance robustness (label-free, no training).

No clean action, no test-time label: the method only transforms the CURRENT
observation, re-runs the frozen policy, and aggregates the action chunks.

Three aggregation families are compared:
  * inverse photometric correction: undo the known color_jitter factors
        (brightness 1.6, contrast 1.6, saturation 2.2 -> multiply by their
         inverses) and run once. This is the input-side compensation route.
  * appearance-variant voting: average / median the action over K photometric
        variants of the same observation (test-time augmentation voting,
        in the spirit of PDF arXiv:2604.18107 and the TTA analyses
        arXiv:2608.17484).
  * no correction baseline.

All conditions share the same fixed flow-matching action noise per state
(upstream action_generator.py:188 redraws randn on every call), so differences
are attributable to the input transform only. Read-only, no gradients.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TVF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "robocasa"))

from dacsc.corp_data import LinearNormalizer, read_episode_meta  # noqa: E402
from dacsc.corp_images import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from dacsc.corp_robocasa import load_adapted_checkpoint  # noqa: E402
from dacsc.eval_protocol import file_sha256  # noqa: E402
from dacsc.upstream import verify_upstream  # noqa: E402
from scripts.eval_corp_robocasa_sr import policy_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", nargs="+", type=int, default=[16, 29, 35, 41, 56, 57, 69, 76])
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 32, 64])
    parser.add_argument("--shift", default="color_jitter")
    parser.add_argument("--sample-steps", type=int, default=None)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def load_episode(dataset: Path, episode: int):
    extra = dataset / "extras" / f"episode_{episode:06d}"
    meta = json.loads((extra / "ep_meta.json").read_text())
    with np.load(extra / "states.npz", allow_pickle=False) as state_file:
        states = np.asarray(state_file["states"])
    with gzip.open(extra / "model.xml.gz", "rb") as model_file:
        model_xml = model_file.read().decode()
    return meta, states, model_xml


MEAN = IMAGENET_MEAN.view(1, 3, 1, 1)
STD = IMAGENET_STD.view(1, 3, 1, 1)


def transform(images: dict[str, torch.Tensor], ops) -> dict[str, torch.Tensor]:
    out = {}
    for name, value in images.items():
        mean = MEAN.to(device=value.device, dtype=value.dtype)
        std = STD.to(device=value.device, dtype=value.dtype)
        rgb = (value * std + mean).clamp(0, 1)
        for op, arg in ops:
            if op == "brightness":
                rgb = TVF.adjust_brightness(rgb, arg)
            elif op == "contrast":
                rgb = TVF.adjust_contrast(rgb, arg)
            elif op == "saturation":
                rgb = TVF.adjust_saturation(rgb, arg)
            elif op == "grayscale":
                rgb = TVF.rgb_to_grayscale(rgb, num_output_channels=3)
            else:
                raise ValueError(f"unknown op {op}")
        out[name] = (rgb - mean) / std
    return out


# inverse of the deterministic color_jitter: brightness 1.6, contrast 1.6, saturation 2.2
INVERSE = [
    ("brightness", 1.0 / 1.6),
    ("contrast", 1.0 / 1.6),
    ("saturation", 1.0 / 2.2),
]
INVERSE_PARTIAL = [
    ("brightness", 0.8),
    ("contrast", 0.8),
    ("saturation", 0.7),
]
VARIANTS = {
    "identity": [],
    "inverse": INVERSE,
    "inverse_partial": INVERSE_PARTIAL,
    "darker": [("brightness", 0.7)],
    "lower_contrast": [("contrast", 0.7)],
    "desaturate": [("saturation", 0.5)],
    "grayscale": [("grayscale", 1.0)],
}


def set_action_noise(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sample(model, images, state, lang, steps, noise_seed):
    set_action_noise(noise_seed)
    with torch.no_grad():
        action = model.sample_action(images, state, lang, num_steps=steps)
    return action.detach().float().flatten()


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.out}")
    upstream = verify_upstream()
    from robocasa.models.scenes.scene_registry import LayoutType
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    metadata = read_episode_meta(args.dataset)
    episode_data = {}
    for episode in args.episodes:
        if episode not in metadata:
            raise ValueError(f"episode is absent: {episode}")
        meta, states, model_xml = load_episode(args.dataset, episode)
        if states.shape[0] <= max(args.frames):
            raise ValueError(f"episode {episode} lacks requested state frames")
        episode_data[episode] = {
            "style": metadata[episode]["style_id"],
            "layout": metadata[episode]["layout_id"],
            "meta": meta,
            "states": states,
            "model_xml": model_xml,
        }

    device = torch.device(args.device)
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    model.eval()
    norm = LinearNormalizer(mean, std)
    styles = sorted({item["style"] for item in episode_data.values()})
    env = RoboCasaGymEnv(
        env_name="OpenDrawer",
        split=None,
        obj_instance_split="pretrain",
        style_ids=styles,
        layout_ids=LayoutType.TRAIN,
        camera_widths=256,
        camera_heights=256,
        seed=args.seed,
    )
    core = env.env
    results = []
    started = time.monotonic()
    try:
        for episode in args.episodes:
            info = episode_data[episode]
            for frame in args.frames:
                reset_to(
                    core,
                    {
                        "model": info["model_xml"],
                        "states": info["states"][frame],
                        "ep_meta": json.dumps(info["meta"]),
                    },
                )
                obs = env.get_observation(core._get_observations(force_update=True))
                images_c, state, lang = policy_inputs(
                    obs, info["meta"]["lang"], norm, device, 0.0, "none"
                )
                images_j, state_j, lang_j = policy_inputs(
                    obs, info["meta"]["lang"], norm, device, 0.0, args.shift
                )
                if not torch.equal(state, state_j) or lang != lang_j:
                    raise RuntimeError("clean and shifted inputs are not paired")
                seed = args.seed * 1000 + len(results)
                action_c = sample(model, images_c, state, lang, args.sample_steps, seed)
                action_j = sample(model, images_j, state, lang, args.sample_steps, seed)
                base_error = (action_j - action_c).norm().clamp_min(1e-8)

                entry = {"episode": episode, "frame": frame, "style": info["style"]}
                jitter_variants = {}
                clean_variants = {}
                for name, ops in VARIANTS.items():
                    jitter_variants[name] = sample(
                        model, transform(images_j, ops), state, lang, args.sample_steps, seed
                    )
                    clean_variants[name] = sample(
                        model, transform(images_c, ops), state, lang, args.sample_steps, seed
                    )
                    entry[f"jitter_{name}_recovery"] = float(
                        (jitter_variants[name] - action_c).norm() / base_error
                    )
                    entry[f"clean_{name}_cost"] = float(
                        (clean_variants[name] - action_c).norm() / base_error
                    )
                stacked = torch.stack(list(jitter_variants.values()))
                mean_action = stacked.mean(dim=0)
                median_action = stacked.median(dim=0).values
                entry["vote_mean_recovery"] = float((mean_action - action_c).norm() / base_error)
                entry["vote_median_recovery"] = float((median_action - action_c).norm() / base_error)
                clean_stacked = torch.stack(list(clean_variants.values()))
                entry["clean_vote_mean_cost"] = float(
                    (clean_stacked.mean(dim=0) - action_c).norm() / base_error
                )
                results.append(entry)
                print(
                    f"ep={episode} f={frame} "
                    f"inv={entry['jitter_inverse_recovery']:.3f} "
                    f"vote_mean={entry['vote_mean_recovery']:.3f} "
                    f"vote_med={entry['vote_median_recovery']:.3f} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

        def summarize(key):
            values = [r[key] for r in results]
            return {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "below_0.8": int(sum(1 for v in values if v < 0.8)),
                "above_1.5": int(sum(1 for v in values if v > 1.5)),
            }

        payload = {
            "schema": "dacsc_tta_voting_probe_v1",
            "method_references": {
                "tta_voting": "PDF arXiv:2604.18107; TTA headroom analysis arXiv:2608.17484",
                "input_normalization": "standard photometric test-time normalization family",
                "paired_flow_noise": "upstream action_generator.py:188",
            },
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "styles": sorted({r["style"] for r in results}),
                "shift": args.shift,
                "states": len(results),
                "variants": list(VARIANTS),
                "uses_clean_action": False,
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "summary": {
                "inverse_recovery": summarize("jitter_inverse_recovery"),
                "inverse_partial_recovery": summarize("jitter_inverse_partial_recovery"),
                "vote_mean_recovery": summarize("vote_mean_recovery"),
                "vote_median_recovery": summarize("vote_median_recovery"),
                "clean_inverse_cost": summarize("clean_inverse_cost"),
                "clean_vote_mean_cost": summarize("clean_vote_mean_cost"),
            },
            "per_variant": {
                name: summarize(f"jitter_{name}_recovery") for name in VARIANTS
            },
            "states": results,
            "status": "completed",
            "elapsed_seconds": None,
        }
        payload["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.out, payload)
        print(json.dumps(payload["summary"], indent=2), flush=True)
        print("DONE ->", args.out, flush=True)
        return 0
    finally:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
