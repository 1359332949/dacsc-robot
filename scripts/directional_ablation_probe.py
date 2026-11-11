"""Directional-ablation probe for appearance directions (read-only, no training).

Grounded in published method + open-source implementation, not invented:

* Directional ablation formula
      h <- h - (h . u_hat) u_hat
  from Arditi et al. 2024, "Refusal in Language Models Is Mediated by a Single
  Direction" (arXiv:2406.11717), and its official implementation
      https://github.com/andyrdt/refusal_direction
      pipeline/utils/hook_utils.py :: get_direction_ablation_output_hook
  (the hook normalises the direction and subtracts the projection on the module
  output; we apply it to a DINO block output, all tokens, exactly as written).

* Appearance directions u_j come from the same-state paired deltas measured by
  lowrank_appearance_probe.py (the "paired render" calibration).

* Evaluation follows the activation-steering convention used by COAST
  (arXiv:2605.17144): does the intervention move the policy output toward the
  clean/reference output, and what does it cost on clean input.

* Matched-budget random directions as control: E-SAE (arXiv:2605.17204) shows a
  score-based result is only credible against a same-size random intervention.

Metrics per state and per direction u_j (a_j^(abl) = sample(f with u_j ablated)):
    recovery   = ||a_j^(abl) - a_clean|| / ||a_jitter - a_clean||   (<1 = fixed)
    clean_cost = ||a_clean^(abl) - a_clean|| / ||a_jitter - a_clean||
    removed    = mean_tokens (f . u_hat)^2  (how much of the feature was removed)
Plus a least-squares decomposition of the real appearance action damage
    a_jitter - a_clean  ~  sum_j c_j (a_clean^(abl_j) - a_clean)
with R^2, i.e. how much of the appearance-induced action change is spanned by
the top-k appearance directions. No gradients, no parameter updates, no rollout.
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
    parser.add_argument("--episodes", nargs="+", type=int, default=[10, 50, 64, 73])
    parser.add_argument(
        "--frames", nargs="+", type=int, default=[0, 16, 32, 48, 64, 80]
    )
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--shift", default="color_jitter")
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


class DirectionalAblation:
    """Block-output hook implementing Arditi et al. directional ablation.

    Mirrors get_direction_ablation_output_hook: normalise u, subtract the
    projection from the activation. When `direction is None` the hook instead
    records the untouched activation for calibration.
    """

    def __init__(self, block):
        self.block = block
        self.direction: torch.Tensor | None = None
        self.capture = False
        self.value: torch.Tensor | None = None
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        if self.capture:
            self.value = output.detach().clone()
            return None
        if self.direction is None:
            return None
        direction = self.direction / (self.direction.norm() + 1e-8)
        direction = direction.to(device=output.device, dtype=output.dtype)
        return output - (output @ direction).unsqueeze(-1) * direction

    def close(self):
        self.handle.remove()


def set_action_noise(seed: int) -> None:
    """Fix the flow-matching initial action noise.

    The upstream sampler starts from torch.randn(...) on every call
    (repos/CoRP/corp/action_generator.py:188), so without this the
    clean/jitter/ablated actions of the SAME state would use different
    noise and every damage/recovery number would be noise-contaminated.
    The official evaluation path also fixes the RNG
    (dacsc/training_state.py:105 evaluation_state).
    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sample(model, images, state, lang, steps, capture=None, noise_seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    if noise_seed is not None:
        set_action_noise(noise_seed)
    if capture is not None:
        capture.capture = True
        capture.value = None
    try:
        with torch.no_grad():
            action = model.sample_action(images, state, lang, num_steps=steps)
    finally:
        if capture is not None:
            capture.capture = False
    feature = None if capture is None else capture.value
    return action.detach().float(), feature


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
    if not 0 <= args.layer < len(backbone.blocks):
        raise ValueError("layer outside the visual encoder depth")
    hook = DirectionalAblation(backbone.blocks[args.layer])
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
                state_index = len(records)
                base_seed = args.seed * 1000 + state_index
                action_c, feature_c = sample(
                    model, images_c, state, lang, args.sample_steps,
                    capture=hook, noise_seed=base_seed,
                )
                action_j, feature_j = sample(
                    model, images_j, state, lang, args.sample_steps,
                    capture=hook, noise_seed=base_seed,
                )
                # noise floor: the same clean input, different action noise
                repeats = []
                for offset in (1, 2, 3):
                    repeated, _ = sample(
                        model, images_c, state, lang, args.sample_steps,
                        capture=None, noise_seed=base_seed + offset * 7919,
                    )
                    repeats.append(repeated.flatten().cpu())
                records.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "style": info["style"],
                        "action_clean": action_c.flatten().cpu(),
                        "action_shift": action_j.flatten().cpu(),
                        "feature_clean": feature_c.mean(dim=(0, 1)).cpu(),
                        "feature_shift": feature_j.mean(dim=(0, 1)).cpu(),
                        "images_c": {k: v.clone() for k, v in images_c.items()},
                        "images_j": {k: v.clone() for k, v in images_j.items()},
                        "state": state.clone(),
                        "lang": list(lang),
                        "noise_seed": base_seed,
                        "noise_repeats": repeats,
                    }
                )
                print(
                    f"paired episode={episode} frame={frame} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

        clean_features = torch.stack([r["feature_clean"] for r in records])
        shift_features = torch.stack([r["feature_shift"] for r in records])
        delta = shift_features - clean_features
        _, _, vh = torch.linalg.svd(delta.double(), full_matrices=False)
        k = min(args.directions, vh.shape[0])
        directions = vh[:k].float()
        generator = torch.Generator().manual_seed(args.seed)
        random_dirs, _ = torch.linalg.qr(
            torch.randn(clean_features.shape[1], args.random_controls, generator=generator)
        )
        random_dirs = random_dirs.T.float()
        all_dirs = [("appearance", i, directions[i]) for i in range(k)] + [
            ("random", i, random_dirs[i]) for i in range(args.random_controls)
        ]

        per_direction = []
        effects: dict[str, list[torch.Tensor]] = {
            f"{kind}{index}": [] for kind, index, _ in all_dirs
        }
        for kind, index, vector in all_dirs:
            vector = vector / vector.norm().clamp_min(1e-8)
            recovery, clean_cost, removed = [], [], []
            for record in records:
                hook.direction = vector
                try:
                    action_j_abl, _ = sample(
                        model,
                        record["images_j"],
                        record["state"],
                        record["lang"],
                        args.sample_steps,
                    )
                    action_c_abl, _ = sample(
                        model,
                        record["images_c"],
                        record["state"],
                        record["lang"],
                        args.sample_steps,
                    )
                finally:
                    hook.direction = None
                base_error = (record["action_shift"] - record["action_clean"]).norm()
                moved = (action_j_abl.flatten().cpu() - record["action_clean"]).norm()
                cost = (action_c_abl.flatten().cpu() - record["action_clean"]).norm()
                recovery.append(float(moved / base_error.clamp_min(1e-8)))
                clean_cost.append(float(cost / base_error.clamp_min(1e-8)))
                removed.append(
                    float(
                        (
                            record["feature_shift"].to(vector.device) @ vector
                        ).pow(2).mean()
                    )
                )
                effects[f"{kind}{index}"].append(
                    (action_c_abl.flatten().cpu() - record["action_clean"]).float()
                )
            per_direction.append(
                {
                    "kind": kind,
                    "index": index,
                    "recovery_mean": float(np.mean(recovery)),
                    "recovery_median": float(np.median(recovery)),
                    "clean_cost_mean": float(np.mean(clean_cost)),
                    "removed_energy_mean": float(np.mean(removed)),
                    "recovery_per_state": recovery,
                }
            )
            print(
                f"ablate kind={kind} index={index} "
                f"recovery={per_direction[-1]['recovery_mean']:.3f} "
                f"clean_cost={per_direction[-1]['clean_cost_mean']:.3f}",
                flush=True,
            )

        # least-squares decomposition of the real action damage
        damage = torch.stack(
            [r["action_shift"] - r["action_clean"] for r in records]
        ).float()
        design = torch.stack(
            [
                torch.stack(effects[f"appearance{i}"])
                for i in range(k)
            ],
            dim=2,
        )  # [states, actions, k]
        r2_values, coefficients = decompose(design, damage)
        # matched-budget control: same in-sample decomposition with random dirs
        design_random = torch.stack(
            [torch.stack(effects[f"random{i}"]) for i in range(args.random_controls)],
            dim=2,
        )
        r2_random, _ = decompose(design_random, damage)
        # noise floor: same clean input, different flow noise
        noise_ratios = []
        for record in records:
            base = (record["action_shift"] - record["action_clean"]).norm().clamp_min(1e-8)
            for repeat in record["noise_repeats"]:
                noise_ratios.append(float((repeat - record["action_clean"]).norm() / base))
        noise_floor = float(np.mean(noise_ratios))
        payload = {
            "schema": "dacsc_directional_ablation_probe_v1",
            "method_references": {
                "directional_ablation": "Arditi et al. 2024, arXiv:2406.11717",
                "directional_ablation_code": (
                    "https://github.com/andyrdt/refusal_direction "
                    "pipeline/utils/hook_utils.py get_direction_ablation_output_hook"
                ),
                "paired_render_calibration": "local lowrank_appearance_probe.py",
                "recovery_convention": "COAST arXiv:2605.17144 (steer toward reference output)",
                "matched_random_control": "E-SAE arXiv:2605.17204 (same-size random intervention)",
                "decomposition": "ordinary least squares, no new statistic",
            },
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
            "directions": per_direction,
            "summary": summarize(per_direction),
            "damage_decomposition": {
                "directions": k,
                "r2_mean": float(np.mean(r2_values)),
                "r2_median": float(np.median(r2_values)),
                "r2_per_state": r2_values,
                "coefficients_mean": [
                    float(v) for v in torch.tensor(coefficients).mean(dim=0)
                ],
            },
            "damage_decomposition_random_control": {
                "directions": int(design_random.shape[2]),
                "r2_mean": float(np.mean(r2_random)),
                "r2_median": float(np.median(r2_random)),
            },
            "action_noise_floor": {
                "definition": "||a(clean, different flow noise) - a(clean)|| / ||a_jitter - a_clean||",
                "mean": noise_floor,
                "note": "if this is >> 0 the earlier unpaired-noise runs were contaminated",
            },
            "status": "completed",
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(args.out, payload)
        print(json.dumps(payload["summary"], indent=2), flush=True)
        print(json.dumps(payload["damage_decomposition"]["r2_mean"], indent=2), flush=True)
        print("DONE ->", args.out, flush=True)
        return 0
    finally:
        hook.close()


def decompose(design: torch.Tensor, damage: torch.Tensor) -> tuple[list[float], list[list[float]]]:
    """Per-state least squares: how much of the action damage lies in the span of the k effect vectors."""
    k = design.shape[2]
    design_flat = design.reshape(damage.shape[0], -1, k)
    damage_flat = damage.reshape(damage.shape[0], -1)
    r2_values: list[float] = []
    coefficients: list[list[float]] = []
    for state_index in range(damage.shape[0]):
        a = design_flat[state_index]
        b = damage_flat[state_index]
        solution = torch.linalg.lstsq(a, b.unsqueeze(-1)).solution.squeeze(-1)
        prediction = a @ solution
        ss_res = float(((b - prediction) ** 2).sum())
        ss_tot = float(((b - b.mean()) ** 2).sum())
        r2_values.append(1.0 - ss_res / max(ss_tot, 1e-12))
        coefficients.append([float(v) for v in solution])
    return r2_values, coefficients


def summarize(per_direction: list[dict]) -> dict:
    appearance = [d for d in per_direction if d["kind"] == "appearance"]
    random_rows = [d for d in per_direction if d["kind"] == "random"]
    out = {
        "appearance_recovery_mean": float(np.mean([d["recovery_mean"] for d in appearance])),
        "appearance_recovery_min": float(np.min([d["recovery_mean"] for d in appearance])),
        "random_recovery_mean": (
            float(np.mean([d["recovery_mean"] for d in random_rows]))
            if random_rows
            else None
        ),
        "appearance_clean_cost_mean": float(
            np.mean([d["clean_cost_mean"] for d in appearance])
        ),
        "random_clean_cost_mean": (
            float(np.mean([d["clean_cost_mean"] for d in random_rows]))
            if random_rows
            else None
        ),
        "spread_across_appearance_directions": float(
            np.max([d["recovery_mean"] for d in appearance])
            - np.min([d["recovery_mean"] for d in appearance])
        ),
    }
    if out["random_recovery_mean"]:
        out["appearance_over_random_recovery"] = (
            out["appearance_recovery_mean"] / out["random_recovery_mean"]
        )
    return out


if __name__ == "__main__":
    raise SystemExit(main())
