"""Per-direction action sensitivity of appearance-shift directions (read-only).

For each restored state:
  * render clean + color_jitter, capture one DINO block, form delta = f_j - f_c;
  * SVD over states -> appearance directions U = [u_1..u_k];
  * push the clean block output along each u_j by eps and re-sample the action
    chunk, giving a_j = ||sample(f + eps*u_j) - sample(f)|| / eps;
  * controls: random orthogonal directions and a between-state direction.

This is the measurement behind "do not remove directions that carry action
information". No gradients, no parameter updates, no training.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", nargs="+", type=int, default=[64, 73])
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 16, 32, 48, 64, 80])
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--shift", default="color_jitter")
    parser.add_argument("--directions", type=int, default=5)
    parser.add_argument("--random-controls", type=int, default=5)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--eps-fraction", type=float, default=0.1)
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


class BlockCapture:
    """Capture one block output; optionally add a fixed vector on replay."""

    def __init__(self, block, layer: int):
        self.block = block
        self.layer = layer
        self.value: torch.Tensor | None = None
        self.vector: torch.Tensor | None = None
        self.handles = []
        self.handles.append(block.register_forward_hook(self._capture))

    def _capture(self, _module, _inputs, output):
        if self.vector is not None:
            return output + self.vector.to(
                device=output.device, dtype=output.dtype
            )
        self.value = output.detach().float().clone()
        return None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


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
    backbone = model.representation_extractor.visual_encoder.backbone
    view_names = tuple(model.view_names)
    if not 0 <= args.layer < len(backbone.blocks):
        raise ValueError("layer outside the visual encoder depth")
    block = backbone.blocks[args.layer]

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
    capture = BlockCapture(block, args.layer)
    records = []
    state_inputs = []
    clean_actions = []
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
                entry = {"episode": episode, "frame": frame, "style": info["style"]}
                observed = {}
                for shift in ("none", args.shift):
                    images, state, lang = policy_inputs(
                        obs, info["meta"]["lang"], norm, device, 0.0, shift
                    )
                    capture.vector = None
                    action = model.sample_action(
                        images, state, lang, num_steps=args.sample_steps
                    )
                    pooled = capture.value.mean(dim=(0, 1)).cpu()
                    entry[shift] = {
                        "feature": pooled.tolist(),
                        "action": action.detach().float().cpu().flatten().tolist(),
                    }
                    observed[shift] = (images, state, lang, action)
                entry["instruction"] = info["meta"]["lang"]
                records.append(entry)
                # keep the clean observation tensors for the direction sweep,
                # so the simulator is never reset twice per state
                state_inputs.append(observed["none"][:3])
                clean_actions.append(
                    observed["none"][3].detach().float().cpu().flatten()
                )
                print(
                    f"pair episode={episode} frame={frame} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

        clean = torch.tensor([r["none"]["feature"] for r in records], dtype=torch.float32)
        shifted = torch.tensor(
            [r[args.shift]["feature"] for r in records], dtype=torch.float32
        )
        delta = shifted - clean
        scale = float(delta.norm(dim=1).mean())
        if scale <= 0:
            raise RuntimeError("appearance delta is degenerate")
        eps = args.eps_fraction * scale

        _, _, vh = torch.linalg.svd(delta.double(), full_matrices=False)
        k = min(args.directions, vh.shape[0])
        directions = vh[:k].float()
        # orthogonal random controls
        generator = torch.Generator().manual_seed(args.seed)
        random_matrix = torch.randn(
            args.random_controls, clean.shape[1], generator=generator
        )
        q, _ = torch.linalg.qr(random_matrix.T)
        random_dirs = q.T[: args.random_controls].float()
        # between-state direction: fixed pair of states, clean features only
        state_dir = (clean[1] - clean[0]).float()
        state_dir = state_dir / state_dir.norm().clamp_min(1e-8)

        result_directions = []
        all_vectors = (
            [("appearance", index, directions[index]) for index in range(k)]
            + [
                ("random", index, random_dirs[index])
                for index in range(random_dirs.shape[0])
            ]
            + [("state", 0, state_dir)]
        )
        payload_phase1 = {
            "schema": "dacsc_action_sensitivity_probe_v1_phase1",
            "scope": {"states": len(records), "layer": args.layer, "shift": args.shift},
            "delta_scale_mean_norm": scale,
            "eps": eps,
            "status": "captured",
        }
        atomic_json(args.out.with_suffix(".phase1.json"), payload_phase1)
        for kind, index, vector in all_vectors:
            vector = vector / vector.norm().clamp_min(1e-8)
            changes = []
            for (images, state, lang), base in zip(state_inputs, clean_actions):
                capture.vector = eps * vector.view(1, 1, -1)
                try:
                    with torch.no_grad():
                        action = model.sample_action(
                            images, state, lang, num_steps=args.sample_steps
                        )
                finally:
                    capture.vector = None
                moved = action.detach().float().cpu().flatten()
                changes.append(float((moved - base).norm()))
            changes_tensor = torch.tensor(changes, dtype=torch.float32)
            result_directions.append(
                {
                    "kind": kind,
                    "index": index,
                    "action_sensitivity": float(changes_tensor.mean() / eps),
                    "action_sensitivity_median": float(changes_tensor.median() / eps),
                    "per_state": [float(v / eps) for v in changes],
                }
            )
            print(
                f"direction kind={kind} index={index} "
                f"sensitivity={result_directions[-1]['action_sensitivity']:.4f}",
                flush=True,
            )

        appearance_sensitivity = float(
            torch.tensor(
                [
                    (
                        torch.tensor(r[args.shift]["action"])
                        - torch.tensor(r["none"]["action"])
                    )
                    .norm()
                    / (torch.tensor(r[args.shift]["feature"]) - torch.tensor(r["none"]["feature"]))
                    .norm()
                    .clamp_min(1e-8)
                    for r in records
                ]
            ).mean()
        )
        appearance_rows = [d for d in result_directions if d["kind"] == "appearance"]
        random_rows = [d for d in result_directions if d["kind"] == "random"]
        state_rows = [d for d in result_directions if d["kind"] == "state"]
        appearance_values = [d["action_sensitivity"] for d in appearance_rows]
        random_values = [d["action_sensitivity"] for d in random_rows]
        payload = {
            "schema": "dacsc_action_sensitivity_probe_v1",
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "layer": args.layer,
                "shift": args.shift,
                "states": len(records),
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "delta_scale_mean_norm": scale,
            "eps": eps,
            "appearance_action_sensitivity_direct": appearance_sensitivity,
            "directions": result_directions,
            "summary": {
                "appearance_mean": float(np.mean(appearance_values)),
                "appearance_max": float(np.max(appearance_values)),
                "appearance_min": float(np.min(appearance_values)),
                "random_mean": float(np.mean(random_values)) if random_values else None,
                "random_max": float(np.max(random_values)) if random_values else None,
                "state_direction": state_rows[0]["action_sensitivity"] if state_rows else None,
                "appearance_over_random": (
                    float(np.mean(appearance_values)) / float(np.mean(random_values))
                    if random_values and mean_nonzero(random_values)
                    else None
                ),
                "spread_across_appearance_directions": float(
                    np.max(appearance_values) - np.min(appearance_values)
                ),
            },
            "status": "completed",
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(args.out, payload)
        print(json.dumps(payload["summary"], indent=2), flush=True)
        print("DONE ->", args.out, flush=True)
        return 0
    finally:
        capture.close()


def mean_nonzero(values: list[float]) -> bool:
    return bool(values) and abs(float(np.mean(values))) > 0


if __name__ == "__main__":
    raise SystemExit(main())
