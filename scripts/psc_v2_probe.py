"""PSC-v2: source-support projection with residual spectral slack calibration.

PSC (arXiv:2603.09338) does two things for source-free regression TTA:
  1) align the target feature inside the SOURCE predictive support, and
  2) calibrate the residual spectral slack in the orthogonal complement.
B2 (tta_adaptation_probe) implemented only (1). This probe adds (2) as a
shrinkage of the orthogonal residual, plus rank/schedule sweeps:

    c  = z - mu
    z' = mu + P c + beta * (c - P c),      P = source top-r subspace

  beta = 0    -> pure projection onto the source support (B2, rank r)
  beta = 0.5  -> keep half of the orthogonal residual (milder calibration)
  beta = 0.75 -> keep three quarters of the residual

Schedules: every Euler step, or the last ceil(n/2) steps (V6 showed late
applications are less harmful on curvature). Leave-one-state-out: the subspace
and mu are fitted on the OTHER states' clean writes only.

Control: random orthonormal subspace with beta=0.

Read-only, no labels, no clean reference at test time.
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
    parser.add_argument("--task", default="OpenDrawer")
    parser.add_argument("--stats-json", type=Path, default=None,
                        help="fixed source stats (mean+basis) from build_psc_stats; disables leave-one-state-out")
    parser.add_argument("--auto-episodes", type=int, default=0,
                        help="if >0, pick this many val-style (51-60) episodes automatically")
    parser.add_argument("--ranks", nargs="+", type=int, default=[3, 6])
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 0.75])
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
    def __init__(self, attention_module):
        self.attention = attention_module
        self.original = attention_module.attend
        self.mode = "idle"
        self.mean: torch.Tensor | None = None
        self.basis: torch.Tensor | None = None
        self.beta = 0.0
        self.target_calls: set[int] | None = None
        self.call = 0
        self.value: torch.Tensor | None = None
        attention_module.attend = self._attend  # type: ignore[assignment]

    def reset(self, mode="idle", mean=None, basis=None, beta=0.0, target_calls=None):
        self.mode = mode
        self.mean = mean
        self.basis = basis
        self.beta = beta
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
        if self.mode == "project" and self.basis is not None:
            if self.target_calls is not None and index not in self.target_calls:
                return output
            mean = self.mean.to(device=output.device, dtype=output.dtype)
            basis = self.basis.to(device=output.device, dtype=output.dtype)
            centred = output - mean
            inside = (centred @ basis.T) @ basis
            outside = centred - inside
            return mean + inside + self.beta * outside
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
    if args.auto_episodes > 0:
        candidates = [
            episode
            for episode, meta in metadata.items()
            if 51 <= int(meta["style_id"]) <= 60
        ]
        args.episodes = candidates[: args.auto_episodes]
        if not args.episodes:
            raise ValueError("no val-style episodes found for auto selection")
    episode_data = {}
    for episode in args.episodes:
        if episode not in metadata:
            raise ValueError(f"episode is absent: {episode}")
        meta, states, model_xml = load_episode(args.dataset, episode)
        if states.shape[0] <= max(args.frames):
            raise ValueError(f"episode {episode} lacks requested state frames")
        episode_data[episode] = {
            "style": metadata[episode]["style_id"],
            "meta": meta,
            "states": states,
            "model_xml": model_xml,
        }

    device = torch.device(args.device)
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    model.eval()
    norm = LinearNormalizer(mean, std)
    hook = WriteHook(model.action_generator.blocks[0].attention)
    steps = args.sample_steps or model.action_generator.num_sample_steps
    late = list(range(math.ceil(steps / 2), steps))
    styles = sorted({item["style"] for item in episode_data.values()})
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
                hook.reset("idle")
                action_j = sample(model, images_j, state, lang, args.sample_steps, seed)
                records.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "style": info["style"],
                        "action_clean": action_c,
                        "action_shift": action_j,
                        "write_clean": write_c,
                        "images_c": {k: v.clone() for k, v in images_c.items()},
                        "images_j": {k: v.clone() for k, v in images_j.items()},
                        "state": state.clone(),
                        "lang": list(lang),
                        "noise_seed": seed,
                    }
                )
                print(f"paired ep={episode} f={frame} elapsed={time.monotonic() - started:.1f}s", flush=True)

        clean_writes = torch.stack([r["write_clean"] for r in records]).float()
        fixed_stats = None
        if args.stats_json is not None:
            payload = json.loads(args.stats_json.read_text())
            fixed_stats = {
                "mean": torch.tensor(payload["mean"], dtype=torch.float32),
                "basis": torch.tensor(payload["basis"], dtype=torch.float32),
            }
        schedules = {"all": set(range(steps)), "late": set(late)}
        rows = {f"r{rank}_b{beta}_{name}": [] for rank in args.ranks for beta in args.betas for name in schedules}
        random_rows = {f"r{rank}": [] for rank in args.ranks}
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        for index, record in enumerate(records):
            base_error = (record["action_shift"] - record["action_clean"]).norm().clamp_min(1e-8)
            train = torch.ones(len(records), dtype=torch.bool)
            train[index] = False
            if fixed_stats is not None:
                source_mean = fixed_stats["mean"]
                vh = fixed_stats["basis"]
            else:
                source = clean_writes[train]
                source_mean = source.mean(dim=0)
                _, _, vh = torch.linalg.svd(source - source_mean, full_matrices=False)
            for rank in args.ranks:
                r = min(rank, vh.shape[0])
                basis = vh[:r].float()
                for beta in args.betas:
                    for name, calls in schedules.items():
                        hook.reset("project", mean=source_mean, basis=basis, beta=beta, target_calls=calls)
                        adapted = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                        hook.reset("idle")
                        rows[f"r{rank}_b{beta}_{name}"].append(
                            float((adapted - record["action_clean"]).norm() / base_error)
                        )
                random_basis = torch.linalg.qr(torch.randn(basis.shape[1], r, generator=generator))[0].T.float()
                hook.reset("project", mean=source_mean, basis=random_basis, beta=0.0, target_calls=schedules["all"])
                adapted = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("idle")
                random_rows[f"r{rank}"].append(float((adapted - record["action_clean"]).norm() / base_error))
            print(f"psc2 ep={record['episode']} f={record['frame']} elapsed={time.monotonic() - started:.1f}s", flush=True)

        def summarize(values):
            return {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "below_0.8": int(sum(1 for v in values if v < 0.8)),
                "above_1.5": int(sum(1 for v in values if v > 1.5)),
            }

        payload = {
            "schema": "dacsc_psc_v2_probe_v1",
            "method_references": {
                "psc": "arXiv:2603.09338 (source support alignment + residual spectral calibration)",
                "scheduling": "per-step vs late-half, after V6",
                "site": "action_generator.blocks[0].attention.attend",
                "paired_flow_noise": "upstream action_generator.py:188",
            },
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "states": len(records),
                "ranks": list(args.ranks),
                "betas": list(args.betas),
                "schedules": list(schedules),
                "leave_one_state_out": fixed_stats is None,
                "stats_json": str(args.stats_json) if args.stats_json else None,
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "summary": {name: summarize(values) for name, values in rows.items()},
            "random": {name: summarize(values) for name, values in random_rows.items()},
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
