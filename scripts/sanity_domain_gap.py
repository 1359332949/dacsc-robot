"""Sanity check: is the appearance shift we probe actually strong, and are we
reading the dataset correctly?

Three conditions on the SAME simulator states:
  clean   : episode 64's own scene (style 47, layout 31, target drawer X)
  jitter  : same scene, deterministic color_jitter applied to the rendered images
  native  : the SAME states replayed with episode 28's scene (style 36, layout 31,
            same target drawer) -> a real native style change, not a photometric filter

Reports, per state:
  * ||a_cond - a_clean|| for jitter and native (action-space domain gap)
  * ||f_cond - f_clean|| at DINO block 8 (feature-space gap)
  * clean action MSE against the dataset's own recorded action (data sanity)
  * pixel mean-abs-diff between clean and each condition (is the scene the same?)

Paired flow-matching action noise (same seed for all three conditions of a state),
because upstream `action_generator.py:188` redraws torch.randn on every call.
Read-only.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "robocasa"))

from dacsc.corp_data import LinearNormalizer, read_episode_meta  # noqa: E402
from dacsc.corp_robocasa import load_adapted_checkpoint  # noqa: E402
from dacsc.eval_protocol import file_sha256  # noqa: E402
from dacsc.upstream import verify_upstream  # noqa: E402
from scripts.eval_corp_robocasa_sr import policy_inputs  # noqa: E402

BASE_EPISODE = 64
NATIVE_EPISODE = 28
DINO_LAYER = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 32, 64, 96, 128, 160])
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


def load_actions(dataset: Path, episode: int) -> np.ndarray:
    import pyarrow.parquet as pq

    path = dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(path, columns=["frame_index", "action"])
    frame_index = np.asarray(table.column("frame_index").to_pylist())
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    order = np.argsort(frame_index)
    return actions[order]


class Capture:
    def __init__(self, module):
        self.value = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.value = output.detach()

    def close(self):
        self.handle.remove()


def set_action_noise(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.out}")
    upstream = verify_upstream()
    from robocasa.models.scenes.scene_registry import LayoutType
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    metadata = read_episode_meta(args.dataset)
    base_meta, base_states, base_xml = load_episode(args.dataset, BASE_EPISODE)
    native_meta, native_states, native_xml = load_episode(args.dataset, NATIVE_EPISODE)
    gt_actions = load_actions(args.dataset, BASE_EPISODE)
    if base_meta["layout_id"] != native_meta["layout_id"]:
        raise ValueError("base and native episodes have different layouts")
    if base_meta.get("object_cfgs") != native_meta.get("object_cfgs"):
        # not fatal: report it, the caller decides
        object_cfg_equal = False
    else:
        object_cfg_equal = True

    device = torch.device(args.device)
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    model.eval()
    norm = LinearNormalizer(mean, std)
    backbone = model.representation_extractor.visual_encoder.backbone
    capture = Capture(backbone.blocks[DINO_LAYER])
    env = RoboCasaGymEnv(
        env_name="OpenDrawer",
        split=None,
        obj_instance_split="pretrain",
        style_ids=sorted({base_meta["style_id"], native_meta["style_id"]}),
        layout_ids=LayoutType.TRAIN,
        camera_widths=256,
        camera_heights=256,
        seed=args.seed,
    )
    core = env.env
    rows = []
    started = time.monotonic()
    try:
        for frame in args.frames:
            if frame >= base_states.shape[0] or frame >= native_states.shape[0]:
                continue
            state_payload = {
                "model": base_xml,
                "states": base_states[frame],
                "ep_meta": json.dumps(base_meta),
            }
            reset_to(core, state_payload)
            obs_clean = env.get_observation(core._get_observations(force_update=True))
            images_clean, state_t, lang = policy_inputs(
                obs_clean, base_meta["lang"], norm, device, 0.0, "none"
            )
            images_jitter, _, _ = policy_inputs(
                obs_clean, base_meta["lang"], norm, device, 0.0, args.shift
            )
            reset_to(
                core,
                {
                    "model": native_xml,
                    "states": base_states[frame],
                    "ep_meta": json.dumps(native_meta),
                },
            )
            obs_native = env.get_observation(core._get_observations(force_update=True))
            images_native, state_n, lang_n = policy_inputs(
                obs_native, native_meta["lang"], norm, device, 0.0, "none"
            )
            if not torch.equal(state_t, state_n):
                raise RuntimeError("state differs between clean and native replay")
            noise_seed = args.seed * 1000 + frame
            actions, features = {}, {}
            for name, images in (
                ("clean", images_clean),
                ("jitter", images_jitter),
                ("native", images_native),
            ):
                set_action_noise(noise_seed)
                capture.value = None
                with torch.no_grad():
                    action = model.sample_action(
                        images, state_t, lang, num_steps=args.sample_steps
                    )
                actions[name] = action.detach().float().flatten().cpu()
                features[name] = capture.value.float().mean(dim=(0, 1)).cpu().squeeze(0)
            pixel_diff = {
                name: float(
                    (
                        images_clean["robot0_agentview_left"]
                        - (
                            images_jitter["robot0_agentview_left"]
                            if name == "jitter"
                            else images_native["robot0_agentview_left"]
                        )
                    ).abs().mean()
                )
                for name in ("jitter", "native")
            }
            gt = torch.as_tensor(gt_actions[min(frame, len(gt_actions) - 1)])
            rows.append(
                {
                    "frame": frame,
                    "style_clean": base_meta["style_id"],
                    "style_native": native_meta["style_id"],
                    "layout": base_meta["layout_id"],
                    "action_delta_jitter": float((actions["jitter"] - actions["clean"]).norm()),
                    "action_delta_native": float((actions["native"] - actions["clean"]).norm()),
                    "feature_delta_jitter": float((features["jitter"] - features["clean"]).norm()),
                    "feature_delta_native": float((features["native"] - features["clean"]).norm()),
                    "clean_action_mse_vs_gt": float(((actions["clean"] - gt) ** 2).mean()),
                    "native_action_mse_vs_gt": float(((actions["native"] - gt) ** 2).mean()),
                    "pixel_mad_jitter": pixel_diff["jitter"],
                    "pixel_mad_native": pixel_diff["native"],
                }
            )
            print(
                f"frame={frame} |da_jit|={rows[-1]['action_delta_jitter']:.4f} "
                f"|da_native|={rows[-1]['action_delta_native']:.4f} "
                f"clean_mse={rows[-1]['clean_action_mse_vs_gt']:.5f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )
        summary = {
            "frames": [r["frame"] for r in rows],
            "action_delta_jitter_mean": float(np.mean([r["action_delta_jitter"] for r in rows])),
            "action_delta_native_mean": float(np.mean([r["action_delta_native"] for r in rows])),
            "feature_delta_jitter_mean": float(np.mean([r["feature_delta_jitter"] for r in rows])),
            "feature_delta_native_mean": float(np.mean([r["feature_delta_native"] for r in rows])),
            "clean_action_mse_vs_gt_mean": float(np.mean([r["clean_action_mse_vs_gt"] for r in rows])),
            "native_action_mse_vs_gt_mean": float(np.mean([r["native_action_mse_vs_gt"] for r in rows])),
            "pixel_mad_jitter_mean": float(np.mean([r["pixel_mad_jitter"] for r in rows])),
            "pixel_mad_native_mean": float(np.mean([r["pixel_mad_native"] for r in rows])),
        }
        payload = {
            "schema": "dacsc_domain_gap_sanity_v1",
            "scope": {
                "dataset": str(args.dataset),
                "base_episode": BASE_EPISODE,
                "base_style": base_meta["style_id"],
                "native_episode": NATIVE_EPISODE,
                "native_style": native_meta["style_id"],
                "layout": base_meta["layout_id"],
                "same_layout": True,
                "object_cfgs_equal": object_cfg_equal,
                "target_drawer_base": base_meta.get("target_drawer"),
                "target_drawer_native": native_meta.get("target_drawer"),
                "shift": args.shift,
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "rows": rows,
            "summary": summary,
            "status": "completed",
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(args.out, payload)
        print(json.dumps(summary, indent=2), flush=True)
        print("DONE ->", args.out, flush=True)
        return 0
    finally:
        capture.close()


if __name__ == "__main__":
    raise SystemExit(main())
