"""Deployable multi-step appearance-subspace projection removal (read-only).

No clean action and no test-time label are used by the method. The only
offline quantity is the appearance direction basis U (top-k of the paired-render
deltas at the cross-attention write), which comes from simulator renders of the
SAME states under different appearances -- no policy rollouts.

At inference each Euler step rewrites the cross-attention write as
    w' = w - sum_j (w . u_j) u_j
(projection onto the appearance subspace removed). Schedules:
    single    : only the last Euler step
    uniform   : every Euler step
    late_half : the last ceil(n/2) Euler steps
Control: the same schedules with k random orthonormal directions.

Grounding: source-statistics alignment / projection is the standard label-free
test-time adaptation family (MixNorm arXiv:2110.11478; source-free regression
TTA arXiv:2603.09338; domain compensation arXiv:2604.21772). Per-step
intervention follows COAST arXiv:2605.17144 and the curvature analysis of
arXiv:2610.16084.
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
    parser.add_argument("--episodes", nargs="+", type=int, default=[16, 29, 35, 41, 56, 57, 69, 76])
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 32, 64])
    parser.add_argument("--shift", default="color_jitter")
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--strength", type=float, default=1.0, help="0..1 partial removal")
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


class WriteHook:
    """Capture or remove a subspace from the cross-attention write."""

    def __init__(self, attention_module):
        self.attention = attention_module
        self.original = attention_module.attend
        self.mode = "idle"          # idle | capture | project_out
        self.basis: torch.Tensor | None = None  # [k, dim]
        self.strength = 1.0
        self.target_calls: set[int] | None = None
        self.call = 0
        self.value: torch.Tensor | None = None
        attention_module.attend = self._attend  # type: ignore[assignment]

    def reset(self, mode="idle", basis=None, strength=1.0, target_calls=None):
        self.mode = mode
        self.basis = basis
        self.strength = strength
        self.target_calls = target_calls
        self.call = 0
        self.value = None

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        index = self.call
        self.call += 1
        if self.mode == "capture":
            self.value = output.detach().clone()
            return output
        if self.mode == "project_out" and self.basis is not None:
            if self.target_calls is not None and index not in self.target_calls:
                return output
            basis = self.basis.to(device=output.device, dtype=output.dtype)
            coefficients = output @ basis.T  # [..., k]
            removed = coefficients @ basis  # [..., dim]
            return output - self.strength * removed
        return output

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


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
    block = model.action_generator.blocks[0]
    hook = WriteHook(block.attention)
    steps = args.sample_steps or model.action_generator.num_sample_steps
    last = steps - 1
    late = list(range(math.ceil(steps / 2), steps))
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
    records = []
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
                seed = args.seed * 1000 + len(records)
                hook.reset("capture")
                action_c = sample(model, images_c, state, lang, args.sample_steps, seed)
                write_c = hook.value.mean(dim=1).squeeze(0).cpu()
                hook.reset("capture")
                action_j = sample(model, images_j, state, lang, args.sample_steps, seed)
                write_j = hook.value.mean(dim=1).squeeze(0).cpu()
                hook.reset("idle")
                records.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "style": info["style"],
                        "action_clean": action_c,
                        "action_shift": action_j,
                        "write_clean": write_c,
                        "write_shift": write_j,
                        "images_c": {k: v.clone() for k, v in images_c.items()},
                        "images_j": {k: v.clone() for k, v in images_j.items()},
                        "state": state.clone(),
                        "lang": list(lang),
                        "noise_seed": seed,
                    }
                )
                print(f"paired ep={episode} f={frame} elapsed={time.monotonic() - started:.1f}s", flush=True)

        clean = torch.stack([r["write_clean"] for r in records]).float()
        shifted = torch.stack([r["write_shift"] for r in records]).float()
        delta = shifted - clean
        _, _, vh = torch.linalg.svd(
            delta.double() - delta.double().mean(0, keepdim=True), full_matrices=False
        )
        k = min(args.candidates, vh.shape[0])
        basis = vh[:k].float()
        generator = torch.Generator().manual_seed(args.seed)
        random_basis, _ = torch.linalg.qr(torch.randn(clean.shape[1], k, generator=generator))
        random_basis = random_basis.T.float()

        schedules = {
            "single": {last},
            "uniform": set(range(steps)),
            "late_half": set(late),
        }
        results = []
        for record in records:
            base_error = (record["action_shift"] - record["action_clean"]).norm().clamp_min(1e-8)
            entry = {"episode": record["episode"], "frame": record["frame"]}
            for name, calls in schedules.items():
                hook.reset("project_out", basis=basis, strength=args.strength, target_calls=calls)
                corrected = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("idle")
                entry[f"{name}_appearance"] = float((corrected - record["action_clean"]).norm() / base_error)
                hook.reset("project_out", basis=random_basis, strength=args.strength, target_calls=calls)
                corrected = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("idle")
                entry[f"{name}_random"] = float((corrected - record["action_clean"]).norm() / base_error)
            results.append(entry)
            print(f"removal ep={record['episode']} f={record['frame']} " + " ".join(f"{k2}={v:.3f}" for k2, v in entry.items() if k2 not in ("episode", "frame")), flush=True)

        def summarize(key):
            values = [r[key] for r in results]
            return {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "below_0.8": int(sum(1 for v in values if v < 0.8)),
                "above_1.5": int(sum(1 for v in values if v > 1.5)),
            }

        payload = {
            "schema": "dacsc_multistep_projection_removal_v1",
            "method_references": {
                "label_free_projection": "MixNorm arXiv:2110.11478; source-free regression TTA arXiv:2603.09338; domain compensation arXiv:2604.21772",
                "site": "cross-attention write, arXiv:2610.20750",
                "per_step": "COAST arXiv:2605.17144; curvature arXiv:2610.16084",
                "paired_flow_noise": "upstream action_generator.py:188",
            },
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "styles": sorted({r["style"] for r in records}),
                "shift": args.shift,
                "site": "action_generator.blocks[0].attention.attend",
                "euler_steps": steps,
                "candidates": k,
                "strength": args.strength,
                "states": len(records),
                "uses_clean_action": False,
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "summary": {
                "single_appearance": summarize("single_appearance"),
                "single_random": summarize("single_random"),
                "uniform_appearance": summarize("uniform_appearance"),
                "uniform_random": summarize("uniform_random"),
                "late_half_appearance": summarize("late_half_appearance"),
                "late_half_random": summarize("late_half_random"),
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
        hook.close()


if __name__ == "__main__":
    raise SystemExit(main())
