"""Read-only clean/color_jitter residual-drift analysis across layer0..11."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "robocasa"))

from dacsc.corp_data import LinearNormalizer, read_episode_meta  # noqa: E402
from dacsc.corp_robocasa import load_adapted_checkpoint  # noqa: E402
from dacsc.eval_protocol import file_sha256  # noqa: E402
from dacsc.upstream import verify_upstream  # noqa: E402
from scripts.eval_corp_robocasa_sr import policy_inputs  # noqa: E402

EPISODES = (10, 50)
FRAMES = (0, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def load_episode(dataset: Path, episode: int) -> tuple[dict, np.ndarray, str]:
    extra = dataset / "extras" / f"episode_{episode:06d}"
    meta = json.loads((extra / "ep_meta.json").read_text())
    with np.load(extra / "states.npz", allow_pickle=False) as state_file:
        states = np.asarray(state_file["states"])
    with gzip.open(extra / "model.xml.gz", "rb") as model_file:
        model_xml = model_file.read().decode()
    return meta, states, model_xml


def capture_layers(backbone, images) -> dict[str, dict[int, torch.Tensor]]:
    captured = {"input": {}, "output": {}}
    hooks = []
    for layer, block in enumerate(backbone.blocks):
        hooks.append(
            block.register_forward_pre_hook(
                lambda _module, inputs, index=layer: captured["input"].__setitem__(
                    index,
                    inputs[0].detach().to(device="cpu", dtype=torch.float32).clone(),
                )
            )
        )
        hooks.append(
            block.register_forward_hook(
                lambda _module, _inputs, output, index=layer: captured[
                    "output"
                ].__setitem__(
                    index,
                    output.detach().to(device="cpu", dtype=torch.float32).clone(),
                )
            )
        )
    try:
        with torch.inference_mode():
            backbone(images)
    finally:
        for hook in hooks:
            hook.remove()
    expected = set(range(len(backbone.blocks)))
    if set(captured["input"]) != expected or set(captured["output"]) != expected:
        raise RuntimeError("incomplete layer residual capture")
    return captured


def compare_tensor(clean: torch.Tensor, jitter: torch.Tensor) -> dict:
    delta = jitter - clean
    clean_rms = clean.pow(2).mean().sqrt()
    jitter_rms = jitter.pow(2).mean().sqrt()
    delta_rms = delta.pow(2).mean().sqrt()
    return {
        "clean_rms": float(clean_rms),
        "jitter_rms": float(jitter_rms),
        "delta_rms": float(delta_rms),
        "relative_rms_diff": float(delta_rms / (clean_rms + 1e-8)),
        "mean_abs_diff": float(delta.abs().mean()),
        "cosine": float(
            F.cosine_similarity(
                clean.flatten().unsqueeze(0),
                jitter.flatten().unsqueeze(0),
                dim=1,
            )[0]
        ),
    }


def layer_metrics(
    captured_clean: dict[str, dict[int, torch.Tensor]],
    captured_jitter: dict[str, dict[int, torch.Tensor]],
    view_names: tuple[str, ...],
) -> list[dict]:
    records = []
    for kind in ("input", "output"):
        for layer in sorted(captured_clean[kind]):
            clean = captured_clean[kind][layer]
            jitter = captured_jitter[kind][layer]
            if clean.shape != jitter.shape or clean.ndim != 3:
                raise ValueError("clean/jitter layer tensors are not aligned")
            if clean.shape[0] != len(view_names) or clean.shape[1] != 257:
                raise ValueError("layer tensor is not [views,CLS+patch,width]")
            for view_index, view in enumerate(view_names):
                for group, start in (("cls", 0), ("patch", 1)):
                    methods = compare_tensor(
                        clean[view_index, start:], jitter[view_index, start:]
                    )
                    records.append(
                        {
                            "layer": layer,
                            "kind": kind,
                            "view": view,
                            "group": group,
                            **{key: round(value, 8) for key, value in methods.items()},
                        }
                    )
    return records


def aggregate(records: list[dict]) -> list[dict]:
    grouped = {}
    for record in records:
        key = (record["layer"], record["kind"], record["view"], record["group"])
        grouped.setdefault(key, []).append(record)
    summary = []
    for (layer, kind, view, group), values in sorted(grouped.items()):
        summary.append(
            {
                "layer": layer,
                "kind": kind,
                "view": view,
                "group": group,
                "mean_relative_rms_diff": round(
                    float(np.mean([value["relative_rms_diff"] for value in values])), 8
                ),
                "max_relative_rms_diff": round(
                    float(np.max([value["relative_rms_diff"] for value in values])), 8
                ),
                "mean_cosine": round(
                    float(np.mean([value["cosine"] for value in values])), 8
                ),
                "min_cosine": round(
                    float(np.min([value["cosine"] for value in values])), 8
                ),
                "observations": len(values),
            }
        )
    return summary


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
    for episode in EPISODES:
        if episode not in metadata:
            raise ValueError(f"episode is absent: {episode}")
        meta, states, model_xml = load_episode(args.dataset, episode)
        if states.shape[0] <= max(FRAMES):
            raise ValueError(f"episode {episode} lacks requested state frames")
        episode_data[episode] = {
            "style": metadata[episode]["style_id"],
            "layout": metadata[episode]["layout_id"],
            "meta": meta,
            "states": states,
            "model_xml": model_xml,
        }

    checkpoint_sha256 = file_sha256(args.model)
    device = torch.device(args.device)
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    norm = LinearNormalizer(mean, std)
    visual = model.representation_extractor.visual_encoder
    backbone = visual.backbone
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
    started = time.monotonic()
    payload = {
        "schema": "dacsc_layer11_residual_drift_v1",
        "started_at_epoch": started,
        "scope": {
            "episodes": list(EPISODES),
            "frames": list(FRAMES),
            "layers": list(range(len(backbone.blocks))),
            "comparison": "clean vs deterministic color_jitter at the same restored simulator state",
            "checkpoint": str(args.model),
            "checkpoint_sha256": checkpoint_sha256,
            "upstream": upstream,
            "read_only": True,
            "action_rollout": False,
        },
        "observations": [],
        "aggregate": [],
        "status": "running",
    }
    atomic_json(args.out, payload)
    all_metrics = []
    try:
        for episode in EPISODES:
            info = episode_data[episode]
            for frame in FRAMES:
                reset_to(
                    core,
                    {
                        "model": info["model_xml"],
                        "states": info["states"][frame],
                        "ep_meta": json.dumps(info["meta"]),
                    },
                )
                obs = env.get_observation(
                    core._get_observations(force_update=True)
                )
                clean_images, state, lang = policy_inputs(
                    obs, info["meta"]["lang"], norm, device, 0.0, "none"
                )
                jitter_images, jitter_state, jitter_lang = policy_inputs(
                    obs, info["meta"]["lang"], norm, device, 0.0, "color_jitter"
                )
                if not torch.equal(state, jitter_state) or lang != jitter_lang:
                    raise RuntimeError("clean and jitter policy inputs are not paired")
                clean_flat = torch.cat(
                    [clean_images[view] for view in model.view_names]
                )
                jitter_flat = torch.cat(
                    [jitter_images[view] for view in model.view_names]
                )
                clean_capture = capture_layers(backbone, clean_flat)
                jitter_capture = capture_layers(backbone, jitter_flat)
                metrics = layer_metrics(
                    clean_capture, jitter_capture, model.view_names
                )
                all_metrics.extend(metrics)
                payload["observations"].append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "style": info["style"],
                        "layout": info["layout"],
                        "metrics": metrics,
                    }
                )
                atomic_json(args.out, payload)
                print(
                    f"captured ep={episode} frame={frame} "
                    f"layers={len(clean_capture['input'])}",
                    flush=True,
                )
        payload["aggregate"] = aggregate(all_metrics)
    finally:
        env.close()
    if file_sha256(args.model) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during residual-drift analysis")
    payload["status"] = "complete"
    payload["elapsed_seconds"] = time.monotonic() - started
    atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "episodes": list(EPISODES),
                "frames": list(FRAMES),
                "observations": len(payload["observations"]),
                "metric_records": len(payload["aggregate"]),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
