"""Low-rank structure of appearance shift in DINO block features (read-only).

For each restored simulator state we render the SAME state under several
photometric appearance shifts, capture every DINO block output, and ask:

  1. how many directions explain the appearance-induced feature delta
     (rank@90, stable rank, participation ratio),
  2. whether those directions are stable across states (split-half principal
     angles),
  3. whether a k-direction subspace explains more delta energy than the best
     k single channels at the same budget (channel vs subspace),
  4. whether different appearance shifts share one subspace (cross-shift
     angles).

No gradients, no parameter updates, no action rollout: forward passes only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "robocasa"))

from dacsc.corp_data import read_episode_meta  # noqa: E402
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
    parser.add_argument("--episodes", nargs="+", type=int, default=[64, 73])
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 16, 32, 48])
    parser.add_argument(
        "--shifts",
        nargs="+",
        default=["grayscale", "color_jitter", "low_contrast", "hue_shift"],
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 2, 5, 8, 11])
    parser.add_argument("--group", choices=["cls", "patch", "all"], default="all")
    parser.add_argument("--rank-k", type=int, default=5)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def load_episode(dataset: Path, episode: int) -> tuple[dict, np.ndarray, str]:
    extra = dataset / "extras" / f"episode_{episode:06d}"
    meta = json.loads((extra / "ep_meta.json").read_text())
    with np.load(extra / "states.npz", allow_pickle=False) as state_file:
        states = np.asarray(state_file["states"])
    with gzip.open(extra / "model.xml.gz", "rb") as model_file:
        model_xml = model_file.read().decode()
    return meta, states, model_xml


@torch.no_grad()
def capture_block_features(
    backbone, flat_images: torch.Tensor, layers: list[int]
) -> dict[int, torch.Tensor]:
    """flat_images: [views, 3, H, W] -> per-layer [views, tokens, C]."""
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            captured[layer] = output.detach().float().clone()

        return hook

    for layer in layers:
        handles.append(backbone.blocks[layer].register_forward_hook(make_hook(layer)))
    try:
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=flat_images.is_cuda
        ):
            backbone(flat_images)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(layers) - set(captured)
    if missing:
        raise RuntimeError(f"DINO blocks were not captured: {sorted(missing)}")
    return captured


def pool_tokens(layer_out: torch.Tensor, group: str) -> torch.Tensor:
    """[views, tokens, C] -> [C]; tokens[0] is CLS, patch tokens follow."""
    if layer_out.ndim != 3:
        raise ValueError("unexpected block output rank")
    if group == "cls":
        selected = layer_out[:, :1, :]
    elif group == "patch":
        selected = layer_out[:, 1:, :]
    else:
        selected = layer_out
    return selected.mean(dim=(0, 1))


def svd_report(delta: torch.Tensor, rank_k: int) -> dict:
    """delta: [N, C]; returns spectrum-based low-rank summary."""
    x = delta.double()
    n = x.shape[0]
    centered = x - x.mean(dim=0, keepdim=True)
    reports = {}
    for name, matrix in (("raw", x), ("centered", centered)):
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        energy = s**2
        total = float(energy.sum())
        if total <= 0:
            reports[name] = {"degenerate": True}
            continue
        ratio = energy / energy.sum()
        cumulative = torch.cumsum(ratio, dim=0)
        rank90 = int(torch.searchsorted(cumulative, torch.tensor(0.90)).item()) + 1
        rank95 = int(torch.searchsorted(cumulative, torch.tensor(0.95)).item()) + 1
        stable_rank = total / float(energy[0]) if float(energy[0]) > 0 else math.nan
        participation = total**2 / float((energy**2).sum())
        k = min(rank_k, len(ratio))
        reports[name] = {
            "n_states": n,
            "total_energy": total,
            "rank90": rank90,
            "rank95": rank95,
            "stable_rank": stable_rank,
            "participation_ratio": participation,
            "top1_ratio": float(ratio[0]),
            "topk_ratio": float(cumulative[k - 1]) if k > 0 else None,
            "topk": k,
            "eigenvalues_top": [float(v) for v in ratio[: min(10, len(ratio))]],
        }
    return reports


def channel_vs_subspace(delta: torch.Tensor, budget: int, centered: bool) -> dict:
    """Energy captured by top-k SVD subspace vs top-k channels at same k."""
    x = delta.double()
    if centered:
        x = x - x.mean(dim=0, keepdim=True)
    total = float((x**2).sum())
    if total <= 0:
        return {"budget": budget, "total_energy": total, "degenerate": True}
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    k = min(budget, vh.shape[0])
    subspace = vh[:k]
    subspace_energy = float(((x @ subspace.T) ** 2).sum())
    channel_energy = (x**2).sum(dim=0)
    top_channels = torch.topk(channel_energy, k).indices
    channel_captured = float((x[:, top_channels] ** 2).sum())
    return {
        "budget": k,
        "budget_fraction": k / x.shape[1],
        "total_energy": total,
        "subspace_ratio": subspace_energy / total,
        "channel_ratio": channel_captured / total,
        "ratio_gain": (subspace_energy + 1e-12) / (channel_captured + 1e-12),
    }


def split_half_angles(delta: torch.Tensor, k: int, seed: int, repeats: int = 8) -> dict:
    """Principal angles between top-k subspaces of two disjoint state halves."""
    x = delta.double()
    n = x.shape[0]
    generator = torch.Generator().manual_seed(seed)
    angles = []
    for _ in range(repeats):
        order = torch.randperm(n, generator=generator)
        half = n // 2
        if half < 2:
            return {"repeats": 0, "insufficient_states": True}
        first, second = order[:half], order[half : 2 * half]
        _, _, vh1 = torch.linalg.svd(x[first], full_matrices=False)
        _, _, vh2 = torch.linalg.svd(x[second], full_matrices=False)
        kk = min(k, vh1.shape[0], vh2.shape[0])
        overlap = vh1[:kk] @ vh2[:kk].T
        cosines = torch.linalg.svdvals(overlap).clamp(-1.0, 1.0)
        deg = torch.rad2deg(torch.arccos(cosines))
        angles.append([float(v) for v in deg])
    array = np.asarray(angles)
    return {
        "repeats": repeats,
        "k": int(min(k, array.shape[1])),
        "mean_deg": [float(v) for v in array.mean(axis=0)],
        "max_deg": [float(v) for v in array.max(axis=0)],
        "worst_over_repeats_deg": [float(v) for v in array.max(axis=0)],
    }


def cross_shift_angles(
    deltas: dict[str, torch.Tensor], k: int, seed: int, repeats: int = 4
) -> dict:
    """Angles between shift-specific top-k subspaces (shared appearance axis?)."""
    names = sorted(deltas)
    pairs = {}
    generator = torch.Generator().manual_seed(seed)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = deltas[names[i]].double(), deltas[names[j]].double()
            n = min(a.shape[0], b.shape[0])
            angles = []
            for _ in range(repeats):
                index = torch.randperm(n, generator=generator)[: max(2, n // 2)]
                _, _, vh1 = torch.linalg.svd(a[index], full_matrices=False)
                _, _, vh2 = torch.linalg.svd(b[index], full_matrices=False)
                kk = min(k, vh1.shape[0], vh2.shape[0])
                overlap = vh1[:kk] @ vh2[:kk].T
                cosines = torch.linalg.svdvals(overlap).clamp(-1.0, 1.0)
                angles.append(
                    [float(v) for v in torch.rad2deg(torch.arccos(cosines))]
                )
            array = np.asarray(angles)
            pairs[f"{names[i]}|{names[j]}"] = {
                "mean_deg": [float(v) for v in array.mean(axis=0)],
                "max_deg": [float(v) for v in array.max(axis=0)],
            }
    return pairs


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.out}")
    if args.group == "all":
        group = "all"
    else:
        group = args.group
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
    from dacsc.corp_data import LinearNormalizer

    norm = LinearNormalizer(mean, std)
    backbone = model.representation_extractor.visual_encoder.backbone
    view_names = tuple(model.view_names)
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

    payload = {
        "schema": "dacsc_lowrank_appearance_probe_v1",
        "scope": {
            "episodes": list(args.episodes),
            "frames": list(args.frames),
            "shifts": ["none"] + list(args.shifts),
            "layers": list(args.layers),
            "token_group": group,
            "comparison": "same restored simulator state, photometric shift after resize",
            "checkpoint": str(args.model),
            "checkpoint_sha256": file_sha256(args.model),
            "upstream": upstream,
            "read_only": True,
            "no_training": True,
            "no_action_rollout": True,
        },
        "states": [],
        "per_layer": {},
        "status": "running",
    }
    atomic_json(args.out, payload)
    features: dict[tuple[int, int, str], dict[int, torch.Tensor]] = {}
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
                for shift in ["none"] + list(args.shifts):
                    images, _state, _lang = policy_inputs(
                        obs, info["meta"]["lang"], norm, device, 0.0, shift
                    )
                    flat = torch.stack([images[name][0] for name in view_names])
                    captured = capture_block_features(backbone, flat, args.layers)
                    pooled = {
                        layer: pool_tokens(captured[layer], group).cpu()
                        for layer in args.layers
                    }
                    features[(episode, frame, shift)] = pooled
                payload["states"].append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "style": info["style"],
                        "layout": info["layout"],
                    }
                )
                print(
                    f"captured episode={episode} frame={frame} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

        n_states = len(payload["states"])
        for layer in args.layers:
            clean = torch.stack(
                [features[(s["episode"], s["frame"], "none")][layer] for s in payload["states"]]
            )
            layer_payload: dict = {
                "states": n_states,
                "clean_energy_mean": float((clean**2).sum(dim=1).mean()),
            }
            deltas: dict[str, torch.Tensor] = {}
            shift_payload = {}
            for shift in args.shifts:
                shifted = torch.stack(
                    [
                        features[(s["episode"], s["frame"], shift)][layer]
                        for s in payload["states"]
                    ]
                )
                delta = shifted - clean
                deltas[shift] = delta
                shift_payload[shift] = {
                    "delta_rms": float(delta.pow(2).mean().sqrt()),
                    "delta_energy_mean": float((delta**2).sum(dim=1).mean()),
                    "relative_delta_energy": float(
                        (delta**2).sum(dim=1).mean() / ((clean**2).sum(dim=1).mean() + 1e-12)
                    ),
                    "cosine_clean_shifted": float(
                        torch.nn.functional.cosine_similarity(
                            clean, shifted, dim=1
                        ).mean()
                    ),
                    "svd": svd_report(delta, args.rank_k),
                    "channel_vs_subspace": {
                        "raw": channel_vs_subspace(delta, 5, centered=False),
                        "raw_5pct": channel_vs_subspace(
                            delta, max(1, round(0.05 * delta.shape[1])), centered=False
                        ),
                        "centered_5pct": channel_vs_subspace(
                            delta, max(1, round(0.05 * delta.shape[1])), centered=True
                        ),
                    },
                    "split_half_angles": split_half_angles(
                        delta, args.rank_k, args.seed
                    ),
                }
            layer_payload["shifts"] = shift_payload
            layer_payload["cross_shift_angles"] = cross_shift_angles(
                deltas, args.rank_k, args.seed
            )
            payload["per_layer"][str(layer)] = layer_payload
            print(f"analyzed layer={layer}", flush=True)
        payload["status"] = "completed"
        payload["elapsed_seconds"] = time.monotonic() - started
        payload["verdict"] = build_verdict(payload)
        atomic_json(args.out, payload)
        print(json.dumps(payload["verdict"], indent=2), flush=True)
        print("DONE ->", args.out, flush=True)
        return 0
    except BaseException:
        payload["status"] = "failed"
        payload["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.out, payload)
        raise


def build_verdict(payload: dict) -> dict:
    """Mechanical read-out of the pre-registered criteria (no cherry-picking)."""
    per_layer = payload.get("per_layer", {})
    best = None
    for layer, data in per_layer.items():
        for shift, stats in data.get("shifts", {}).items():
            centered = stats["svd"].get("centered", {})
            if centered.get("degenerate"):
                continue
            rank90 = centered.get("rank90")
            angles = stats.get("split_half_angles", {})
            max_angle = max(angles.get("mean_deg", [180.0]))
            gain = stats["channel_vs_subspace"]["raw_5pct"].get("ratio_gain", 0.0)
            score = (
                float(rank90 or 999) + float(max_angle) / 10.0 - float(gain)
            )
            candidate = {
                "layer": int(layer),
                "shift": shift,
                "rank90_centered": rank90,
                "split_half_max_mean_angle_deg": max_angle,
                "subspace_over_channel_5pct": gain,
                "score": score,
            }
            if best is None or score < best["score"]:
                best = candidate
    verdict = {
        "pre_registered_criteria": {
            "low_rank": "rank90_centered <= 8",
            "stable": "split-half max mean principal angle <= 20 deg",
            "unit_upgrade": "subspace/channel energy ratio at 5% budget >= 1.2",
        },
        "best_layer_shift": best,
        "note": "mechanical argmin over (rank + angle/10 - gain); inspect all layers before deciding",
    }
    if best:
        verdict["passes"] = {
            "low_rank": (best["rank90_centered"] or 999) <= 8,
            "stable": best["split_half_max_mean_angle_deg"] <= 20.0,
            "unit_upgrade": best["subspace_over_channel_5pct"] >= 1.2,
        }
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
