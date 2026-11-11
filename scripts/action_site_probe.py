"""Same-state paired appearance probe at the ACTION-SIDE sites (read-only).

Motivation (literature, not invention):
* E-SAE (arXiv:2605.17204) measures that single-feature edits on the PaliGemma
  vision-language backbone move success by -0.7..-2.8 pts (= random noise),
  while edits on the action expert move it by -81..-96 pts.
* COAST (arXiv:2605.17144) steers the action expert's residual stream, per
  denoising step, and reports +0.17 success on RoboCasa/pi0.5.
So the same test that failed on the DINO block output is repeated here on
  (a) the resampler output  -> the compressed visual summary that enters the
      action expert, and
  (b) the DiT block output for action tokens -> the action-expert residual
      stream, one chosen Euler step (COAST-style per-step intervention).

Method pieces are reused unchanged from the earlier probes:
* directional ablation  h <- h - (h.u)u   (Arditi et al. 2024, arXiv:2406.11717;
  code github.com/andyrdt/refusal_direction pipeline/utils/hook_utils.py)
* recovery = ||a_ablated - a_clean|| / ||a_jitter - a_clean||  (COAST convention)
* matched random directions (E-SAE)
* least-squares damage decomposition (ordinary least squares)

No gradients, no parameter updates, no training, no rollout.
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
    parser.add_argument("--site", choices=["resampler", "dit"], default="resampler")
    parser.add_argument("--dit-step", type=int, default=-1)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--random-controls", type=int, default=3)
    parser.add_argument("--sample-steps", type=int, default=None)
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


class SiteHook:
    """Capture or directionally ablate a module output.

    mode:
      idle     -> pass through
      capture  -> store detached outputs, indexed by call index
      ablate   -> subtract the projection onto `direction` at `target_step`
                  (target_step=None means every call; the resampler is called
                  once per sample, the DiT block once per Euler step)
    """

    def __init__(self, module):
        self.module = module
        self.mode = "idle"
        self.direction: torch.Tensor | None = None
        self.target_step: int | None = None
        self.call = 0
        self.captured: dict[int, torch.Tensor] = {}
        self.handle = module.register_forward_hook(self._hook)

    def reset(self, mode: str = "idle", direction=None, target_step=None):
        self.mode = mode
        self.direction = direction
        self.target_step = target_step
        self.call = 0
        self.captured = {}

    def _hook(self, _module, _inputs, output):
        index = self.call
        self.call += 1
        if isinstance(output, tuple):
            output = output[0]
        if self.mode == "capture":
            self.captured[index] = output.detach().clone()
            return None
        if self.mode == "ablate":
            if self.target_step is not None and index != self.target_step:
                return None
            direction = self.direction
            if direction is None:
                return None
            direction = direction / (direction.norm() + 1e-8)
            direction = direction.to(device=output.device, dtype=output.dtype)
            return output - (output @ direction).unsqueeze(-1) * direction
        return None

    def close(self):
        self.handle.remove()


def sample(model, images, state, lang, steps):
    with torch.no_grad():
        action = model.sample_action(images, state, lang, num_steps=steps)
    return action.detach().float()


def pooled_at(hook: SiteHook, call_index: int | None) -> torch.Tensor:
    if call_index is None:
        keys = sorted(hook.captured)
        index = keys[-1]
    else:
        index = call_index
    value = hook.captured[index].float()
    return value.mean(dim=1)  # [B, dim]


def svd_summary(delta: torch.Tensor, k: int) -> dict:
    x = delta.double()
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    energy = s**2
    ratio = energy / energy.sum()
    cumulative = torch.cumsum(ratio, 0)
    rank90 = int(torch.searchsorted(cumulative, torch.tensor(0.90)).item()) + 1
    stable = float(energy.sum() / energy[0])
    budget = max(1, round(0.05 * x.shape[1]))
    sub = float(cumulative[min(budget, len(cumulative)) - 1])
    channel_energy = (x**2).sum(dim=0)
    top_channels = torch.topk(channel_energy, min(budget, x.shape[1])).indices
    chan = float((x[:, top_channels] ** 2).sum() / (x**2).sum())
    return {
        "rank90": rank90,
        "stable_rank": stable,
        "top5_ratio": float(cumulative[min(k, len(cumulative)) - 1]),
        "budget_fraction": 0.05,
        "subspace_ratio_5pct": sub,
        "channel_ratio_5pct": chan,
        "subspace_over_channel_5pct": sub / max(chan, 1e-12),
        "directions": vh[: min(k, vh.shape[0])].float().cpu(),
    }


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
    if args.site == "resampler":
        module = model.representation_extractor.cross_attention_resampler
        step_index = None
    else:
        if not model.action_generator.blocks:
            raise ValueError("action generator has no DiT blocks")
        module = model.action_generator.blocks[0]
        steps = args.sample_steps or model.action_generator.num_sample_steps
        step_index = args.dit_step if args.dit_step >= 0 else steps - 1
    hook = SiteHook(module)
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
                hook.reset("capture")
                action_c = sample(model, images_c, state, lang, args.sample_steps)
                feature_c = pooled_at(hook, step_index).cpu()
                hook.reset("capture")
                action_j = sample(model, images_j, state, lang, args.sample_steps)
                feature_j = pooled_at(hook, step_index).cpu()
                hook.reset("idle")
                records.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "style": info["style"],
                        "action_clean": action_c.flatten().cpu(),
                        "action_shift": action_j.flatten().cpu(),
                        "feature_clean": feature_c.squeeze(0),
                        "feature_shift": feature_j.squeeze(0),
                        "images_c": {k: v.clone() for k, v in images_c.items()},
                        "images_j": {k: v.clone() for k, v in images_j.items()},
                        "state": state.clone(),
                        "lang": list(lang),
                    }
                )
                print(
                    f"paired episode={episode} frame={frame} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

        clean = torch.stack([r["feature_clean"] for r in records]).float()
        shifted = torch.stack([r["feature_shift"] for r in records]).float()
        delta = shifted - clean
        spectrum = svd_summary(delta, args.directions)
        directions = spectrum.pop("directions")
        k = directions.shape[0]
        generator = torch.Generator().manual_seed(args.seed)
        random_dirs, _ = torch.linalg.qr(
            torch.randn(clean.shape[1], args.random_controls, generator=generator)
        )
        random_dirs = random_dirs.T.float()
        all_dirs = [("appearance", i, directions[i]) for i in range(k)] + [
            ("random", i, random_dirs[i]) for i in range(args.random_controls)
        ]

        per_direction = []
        effects: dict[str, list[torch.Tensor]] = {}
        for kind, index, vector in all_dirs:
            vector = vector / vector.norm().clamp_min(1e-8)
            recovery, clean_cost = [], []
            state_effects: list[torch.Tensor] = []
            for record in records:
                hook.reset("ablate", direction=vector, target_step=step_index)
                try:
                    action_j_abl = sample(
                        model, record["images_j"], record["state"], record["lang"], args.sample_steps
                    )
                    action_c_abl = sample(
                        model, record["images_c"], record["state"], record["lang"], args.sample_steps
                    )
                finally:
                    hook.reset("idle")
                base_error = (record["action_shift"] - record["action_clean"]).norm()
                moved = (action_j_abl.flatten().cpu() - record["action_clean"]).norm()
                cost = (action_c_abl.flatten().cpu() - record["action_clean"]).norm()
                recovery.append(float(moved / base_error.clamp_min(1e-8)))
                clean_cost.append(float(cost / base_error.clamp_min(1e-8)))
                state_effects.append(
                    (action_c_abl.flatten().cpu() - record["action_clean"]).float()
                )
            effects[f"{kind}{index}"] = state_effects
            per_direction.append(
                {
                    "kind": kind,
                    "index": index,
                    "recovery_mean": float(np.mean(recovery)),
                    "clean_cost_mean": float(np.mean(clean_cost)),
                    "recovery_per_state": recovery,
                }
            )
            print(
                f"ablate kind={kind} index={index} "
                f"recovery={per_direction[-1]['recovery_mean']:.3f} "
                f"clean_cost={per_direction[-1]['clean_cost_mean']:.3f}",
                flush=True,
            )

        damage = torch.stack(
            [r["action_shift"] - r["action_clean"] for r in records]
        ).float()
        design = torch.stack(
            [torch.stack(effects[f"appearance{i}"]) for i in range(k)], dim=2
        ).reshape(damage.shape[0], -1, k)
        damage_flat = damage.reshape(damage.shape[0], -1)
        r2_values, coefficients = [], []
        for state_index in range(damage.shape[0]):
            a = design[state_index]
            b = damage_flat[state_index]
            solution = torch.linalg.lstsq(a, b.unsqueeze(-1)).solution.squeeze(-1)
            prediction = a @ solution
            ss_res = float(((b - prediction) ** 2).sum())
            ss_tot = float(((b - b.mean()) ** 2).sum())
            r2_values.append(1.0 - ss_res / max(ss_tot, 1e-12))
            coefficients.append([float(v) for v in solution])

        payload = {
            "schema": "dacsc_action_site_probe_v1",
            "method_references": {
                "site_choice": "E-SAE arXiv:2605.17204 (action expert has leverage, vision backbone does not); COAST arXiv:2605.17144 (action-expert residual steering)",
                "directional_ablation": "Arditi et al. 2024 arXiv:2406.11717; github.com/andyrdt/refusal_direction pipeline/utils/hook_utils.py",
                "recovery_convention": "COAST arXiv:2605.17144",
                "matched_random": "E-SAE arXiv:2605.17204",
                "decomposition": "ordinary least squares",
            },
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "styles": sorted({r["style"] for r in records}),
                "shift": args.shift,
                "site": args.site,
                "module": type(module).__name__,
                "dit_step": step_index,
                "states": len(records),
                "feature_dim": int(clean.shape[1]),
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "delta_spectrum": spectrum,
            "directions": per_direction,
            "summary": summarize(per_direction),
            "damage_decomposition": {
                "directions": k,
                "r2_mean": float(np.mean(r2_values)),
                "r2_median": float(np.median(r2_values)),
                "r2_per_state": r2_values,
                "coefficients_mean": [float(v) for v in torch.tensor(coefficients).mean(0)],
            },
            "status": "completed",
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(args.out, payload)
        print(json.dumps(payload["summary"], indent=2), flush=True)
        print("R2:", payload["damage_decomposition"]["r2_mean"], flush=True)
        print("DONE ->", args.out, flush=True)
        return 0
    finally:
        hook.close()


def summarize(per_direction: list[dict]) -> dict:
    appearance = [d for d in per_direction if d["kind"] == "appearance"]
    random_rows = [d for d in per_direction if d["kind"] == "random"]
    out = {
        "appearance_recovery_mean": float(np.mean([d["recovery_mean"] for d in appearance])),
        "appearance_recovery_min": float(np.min([d["recovery_mean"] for d in appearance])),
        "random_recovery_mean": (
            float(np.mean([d["recovery_mean"] for d in random_rows])) if random_rows else None
        ),
        "appearance_clean_cost_mean": float(
            np.mean([d["clean_cost_mean"] for d in appearance])
        ),
        "random_clean_cost_mean": (
            float(np.mean([d["clean_cost_mean"] for d in random_rows])) if random_rows else None
        ),
    }
    if out["random_recovery_mean"]:
        out["appearance_over_random_recovery"] = (
            out["appearance_recovery_mean"] / out["random_recovery_mean"]
        )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
