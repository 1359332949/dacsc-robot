"""Multi-step oracle correction at the cross-attention interface (read-only).

Follow-up to the single-shot oracle test: that run could cancel ~half the action
damage in the least-squares sense (residual ratio 0.499) but overshot on 4/24
states (recovery > 1.5), which is the "sharp directions / curvature" failure
described by arXiv:2610.16084.

This probe tests whether spreading the SAME correction across several Euler
steps removes the overshoot, and compares three application schedules:

  single   : the whole push at the last Euler step (reproduction of V5)
  uniform  : push / n_steps at every Euler step
  late_half: push / ceil(n/2) on the last ceil(n/2) steps

All conditions share the same fixed flow-matching action noise per state
(upstream action_generator.py:188 redraws randn on every call).
The correction coefficients come from the single-step least squares, i.e. this
is still an oracle (it uses the clean action), not a deployable method.
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
    parser.add_argument("--random-controls", type=int, default=3)
    parser.add_argument("--eps-fraction", type=float, default=0.1)
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
    """Capture / push on the cross-attention write (SelfAttention.attend output)."""

    def __init__(self, attention_module):
        self.attention = attention_module
        self.original = attention_module.attend
        self.mode = "idle"
        self.vector: torch.Tensor | None = None
        self.target_calls: set[int] | None = None
        self.call = 0
        self.value: torch.Tensor | None = None
        attention_module.attend = self._attend  # type: ignore[assignment]

    def reset(self, mode="idle", vector=None, target_calls=None):
        self.mode = mode
        self.vector = vector
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
        if self.mode == "add" and self.vector is not None:
            if self.target_calls is not None and index not in self.target_calls:
                return output
            return output + self.vector.to(device=output.device, dtype=output.dtype)
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
                index = len(records)
                base_seed = args.seed * 1000 + index
                hook.reset("capture")
                action_c = sample(model, images_c, state, lang, args.sample_steps, base_seed)
                write_c = hook.value.mean(dim=1).squeeze(0).cpu()
                hook.reset("capture")
                action_j = sample(model, images_j, state, lang, args.sample_steps, base_seed)
                write_j = hook.value.mean(dim=1).squeeze(0).cpu()
                hook.reset("idle")
                records.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "action_clean": action_c,
                        "action_shift": action_j,
                        "write_clean": write_c,
                        "write_shift": write_j,
                        "images_j": {k: v.clone() for k, v in images_j.items()},
                        "state": state.clone(),
                        "lang": list(lang),
                        "noise_seed": base_seed,
                    }
                )
                print(f"paired ep={episode} f={frame} elapsed={time.monotonic() - started:.1f}s", flush=True)

        clean = torch.stack([r["write_clean"] for r in records]).float()
        shifted = torch.stack([r["write_shift"] for r in records]).float()
        delta = shifted - clean
        _, s, vh = torch.linalg.svd(
            delta.double() - delta.double().mean(0, keepdim=True), full_matrices=False
        )
        k = min(args.candidates, vh.shape[0])
        candidates = vh[:k].float()
        scale = float(delta.norm(dim=1).mean())
        eps = args.eps_fraction * scale
        generator = torch.Generator().manual_seed(args.seed)
        random_dirs, _ = torch.linalg.qr(
            torch.randn(clean.shape[1], args.random_controls, generator=generator)
        )
        random_dirs = random_dirs.T.float()

        states_out = []
        for record in records:
            device_a = record["action_clean"].device
            dim = record["action_clean"].numel()
            effects = torch.zeros(dim, k, dtype=torch.float32, device=device_a)
            for j in range(k):
                hook.reset("add", vector=eps * candidates[j], target_calls={last})
                pushed = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("idle")
                effects[:, j] = pushed - record["action_shift"]
            target = record["action_clean"] - record["action_shift"]
            solution = torch.linalg.lstsq(effects.double(), target.double().unsqueeze(-1)).solution.squeeze(-1)
            coef = solution.detach().float().cpu()
            residual = float((effects.double() @ solution - target.double()).norm() / target.double().norm().clamp_min(1e-8))
            combined = eps * (candidates * coef.unsqueeze(1)).sum(dim=0)
            base_error = target.norm().clamp_min(1e-8)

            schedule = {}
            # single-shot (reproduction)
            hook.reset("add", vector=combined, target_calls={last})
            corrected = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
            hook.reset("idle")
            schedule["single"] = float((corrected - record["action_clean"]).norm() / base_error)
            # uniform over all steps
            hook.reset("add", vector=combined / steps, target_calls=set(range(steps)))
            corrected = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
            hook.reset("idle")
            schedule["uniform"] = float((corrected - record["action_clean"]).norm() / base_error)
            # late half
            hook.reset("add", vector=combined / len(late), target_calls=set(late))
            corrected = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
            hook.reset("idle")
            schedule["late_half"] = float((corrected - record["action_clean"]).norm() / base_error)
            states_out.append(
                {
                    "episode": record["episode"],
                    "frame": record["frame"],
                    "coefficients": [float(v) for v in coef],
                    "oracle_residual_ratio": residual,
                    "schedule_recovery": schedule,
                    "damage_norm": float(base_error),
                }
            )
            print(
                f"multi ep={record['episode']} f={record['frame']} residual={residual:.3f} "
                f"single={schedule['single']:.3f} uniform={schedule['uniform']:.3f} "
                f"late_half={schedule['late_half']:.3f}",
                flush=True,
            )

        def summarize(key):
            values = [r["schedule_recovery"][key] for r in states_out]
            return {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "below_0.8": int(sum(1 for v in values if v < 0.8)),
                "above_1.5": int(sum(1 for v in values if v > 1.5)),
            }

        payload = {
            "schema": "dacsc_multistep_oracle_correction_v1",
            "method_references": {
                "site": "cross-attention write, arXiv:2610.20750",
                "curvature / overshoot": "arXiv:2610.16084 (sharp directions)",
                "per-step steering": "COAST arXiv:2605.17144",
                "paired_flow_noise": "upstream action_generator.py:188",
            },
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "shift": args.shift,
                "site": "action_generator.blocks[0].attention.attend",
                "euler_steps": steps,
                "late_steps": late,
                "states": len(records),
                "candidates": k,
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "summary": {
                "oracle_residual_ratio_mean": float(np.mean([r["oracle_residual_ratio"] for r in states_out])),
                "single": summarize("single"),
                "uniform": summarize("uniform"),
                "late_half": summarize("late_half"),
            },
            "states": states_out,
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
