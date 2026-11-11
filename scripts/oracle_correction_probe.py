"""Oracle linear correction at the cross-attention interface (read-only).

The V4 probe showed the interface is causally live (appearance-direction
attribution ~10x random) but *removing* those directions does not fix the
action. This probe asks the decisive follow-up:

    is the appearance-induced action damage linearly correctable at this
    interface at all, when we are allowed to choose the push (oracle)?

For each state:
  * e_j = a(push +eps*u_j) - a(jitter)      (action effect of each candidate)
  * target correction t = a(clean) - a(jitter)
  * solve  min_c || sum_j c_j e_j - t ||    (ordinary least squares, oracle)
  * apply the combined push v = eps * sum_j c_j u_j and measure
        recovery = ||a(corrected) - a(clean)|| / ||a(jitter) - a(clean)||
  * controls: same coefficients on random directions; no correction (=1).
  * also report the residual ratio ||t - Ec|| / ||t|| = the best achievable
    linear cancellation with k pushes, independent of how we apply it.

All counterfactuals share the same fixed flow-matching action noise.
References: site = action-conditioning interface (arXiv:2610.20750);
intervention-attribution selection (arXiv:2610.15850); directional-ablation
formula for the controls (Arditi et al. 2024).
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
    parser.add_argument("--episodes", nargs="+", type=int, default=[16, 29, 35, 41, 56, 57, 69, 76])
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 32, 64])
    parser.add_argument("--shift", default="color_jitter")
    parser.add_argument("--step", type=int, default=-1)
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
    """Capture or add a vector to the cross-attention write (SelfAttention.attend output)."""

    def __init__(self, attention_module):
        self.attention = attention_module
        self.original = attention_module.attend
        self.mode = "idle"
        self.vector: torch.Tensor | None = None
        self.target_call: int | None = None
        self.call = 0
        self.value: torch.Tensor | None = None
        attention_module.attend = self._attend  # type: ignore[assignment]

    def reset(self, mode="idle", vector=None, target_call=None):
        self.mode = mode
        self.vector = vector
        self.target_call = target_call
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
            if self.target_call is not None and index != self.target_call:
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
    step_index = args.step if args.step >= 0 else steps - 1
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
                hook.reset("capture", target_call=step_index)
                action_c = sample(model, images_c, state, lang, args.sample_steps, base_seed)
                write_c = hook.value.mean(dim=1).cpu() if hook.value is not None else None
                hook.reset("capture", target_call=step_index)
                action_j = sample(model, images_j, state, lang, args.sample_steps, base_seed)
                write_j = hook.value.mean(dim=1).cpu() if hook.value is not None else None
                hook.reset("idle")
                if write_c is None or write_j is None:
                    raise RuntimeError("cross-attention write was not captured")
                records.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "style": info["style"],
                        "action_clean": action_c,
                        "action_shift": action_j,
                        "write_clean": write_c.squeeze(0),
                        "write_shift": write_j.squeeze(0),
                        "images_c": {k: v.clone() for k, v in images_c.items()},
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
            dim = record["action_clean"].numel()
            target_device = record["action_clean"].device
            effects = torch.zeros(dim, k, dtype=torch.float32, device=target_device)
            for j in range(k):
                hook.reset("add", vector=eps * candidates[j], target_call=step_index)
                pushed = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("idle")
                effects[:, j] = pushed - record["action_shift"]
            target = record["action_clean"] - record["action_shift"]
            solution = torch.linalg.lstsq(effects.double(), target.double().unsqueeze(-1)).solution.squeeze(-1)
            residual = (effects.double() @ solution - target.double()).norm() / target.double().norm().clamp_min(1e-8)
            combined = eps * (candidates * solution.detach().float().cpu().unsqueeze(1)).sum(dim=0)
            hook.reset("add", vector=combined, target_call=step_index)
            corrected = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
            hook.reset("idle")
            recovery = float((corrected - record["action_clean"]).norm() / (record["action_shift"] - record["action_clean"]).norm().clamp_min(1e-8))

            random_effects = torch.zeros(dim, random_dirs.shape[0], dtype=torch.float32, device=target_device)
            for j in range(random_dirs.shape[0]):
                hook.reset("add", vector=eps * random_dirs[j], target_call=step_index)
                pushed = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("idle")
                random_effects[:, j] = pushed - record["action_shift"]
            random_solution = torch.linalg.lstsq(random_effects.double(), target.double().unsqueeze(-1)).solution.squeeze(-1)
            random_residual = (random_effects.double() @ random_solution - target.double()).norm() / target.double().norm().clamp_min(1e-8)
            random_combined = eps * (random_dirs * random_solution.detach().float().cpu().unsqueeze(1)).sum(dim=0)
            hook.reset("add", vector=random_combined, target_call=step_index)
            random_corrected = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
            hook.reset("idle")
            random_recovery = float((random_corrected - record["action_clean"]).norm() / (record["action_shift"] - record["action_clean"]).norm().clamp_min(1e-8))

            states_out.append(
                {
                    "episode": record["episode"],
                    "frame": record["frame"],
                    "coefficients": [float(v) for v in solution],
                    "oracle_residual_ratio": float(residual),
                    "oracle_recovery": recovery,
                    "random_residual_ratio": float(random_residual),
                    "random_recovery": random_recovery,
                    "damage_norm": float((record["action_shift"] - record["action_clean"]).norm()),
                }
            )
            print(
                f"oracle ep={record['episode']} f={record['frame']} "
                f"residual={float(residual):.3f} recovery={recovery:.3f} "
                f"(random residual={float(random_residual):.3f} recovery={random_recovery:.3f})",
                flush=True,
            )

        payload = {
            "schema": "dacsc_oracle_correction_probe_v1",
            "method_references": {
                "site": "cross-attention write = action-conditioning interface, arXiv:2610.20750",
                "attribution": "small-step steering, arXiv:2610.15850",
                "paired_flow_noise": "upstream action_generator.py:188",
            },
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "styles": sorted({r["style"] for r in records}),
                "shift": args.shift,
                "site": "action_generator.blocks[0].attention.attend",
                "euler_step": step_index,
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
                "oracle_recovery_mean": float(np.mean([r["oracle_recovery"] for r in states_out])),
                "oracle_recovery_median": float(np.median([r["oracle_recovery"] for r in states_out])),
                "random_residual_ratio_mean": float(np.mean([r["random_residual_ratio"] for r in states_out])),
                "random_recovery_mean": float(np.mean([r["random_recovery"] for r in states_out])),
                "damage_norm_mean": float(np.mean([r["damage_norm"] for r in states_out])),
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
