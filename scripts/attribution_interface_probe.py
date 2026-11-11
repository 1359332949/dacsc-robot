"""Intervention-attribution direction selection at CoRP's cross-attention write.

Correction of the earlier probes, following the 2026 mechanistic-interpretability
line rather than the channel-dropout line:

* site: the cross-attention WRITE of the single DiT block (action tokens ->
  prefix), which is the analogue of the "action-conditioning interface" found by
  activation patching in pi0.5 / MolmoAct2 / MolmoBot (arXiv:2610.20750).
* direction selection: INTERVENTION ATTRIBUTION, i.e. the small-step response
  || action(f + eps*u) - action(f) || / eps, not the appearance variance.
  This is what arXiv:2610.15850 shows to be the right estimand: observational
  feature-discovery scores do not identify intervention attribution, while
  small-step steering does.
* comparison: for the same candidate directions, measure
  (a) attribution a_j (push response) and (b) removal recovery r_j, then check
  whether a_j ranks the useful removals; compare top-by-attribution vs
  top-by-variance (PCA order) vs random directions.
* every counterfactual (clean / jitter / push / remove) uses the SAME fixed
  flow-matching action noise for a given state (the earlier probes missed this;
  upstream `action_generator.py:188` redraws torch.randn on every call).

Read-only: no gradients through the policy, no parameter updates, no training.
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
    parser.add_argument("--step", type=int, default=-1, help="Euler step to intervene on (-1 = last)")
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


def load_episode(dataset: Path, episode: int) -> tuple[dict, np.ndarray, str]:
    extra = dataset / "extras" / f"episode_{episode:06d}"
    meta = json.loads((extra / "ep_meta.json").read_text())
    with np.load(extra / "states.npz", allow_pickle=False) as state_file:
        states = np.asarray(state_file["states"])
    with gzip.open(extra / "model.xml.gz", "rb") as model_file:
        model_xml = model_file.read().decode()
    return meta, states, model_xml


class AttendHook:
    """Wrap SelfAttention.attend: capture, push, or directionally ablate the write.

    Mirrors the causal-interface idea of arXiv:2610.20750 (patch the
    cross-attention write) and the directional-ablation formula of
    Arditi et al. 2024 (pipeline/utils/hook_utils.py).
    """

    def __init__(self, attention_module):
        self.attention = attention_module
        self.original = attention_module.attend
        self.mode = "idle"          # idle | capture | push | ablate
        self.vector: torch.Tensor | None = None
        self.eps = 0.0
        self.target_call: int | None = None
        self.call = 0
        self.value: torch.Tensor | None = None
        attention_module.attend = self._attend  # type: ignore[assignment]

    def reset(self, mode="idle", vector=None, eps=0.0, target_call=None):
        self.mode = mode
        self.vector = vector
        self.eps = eps
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
        if self.vector is None or (self.target_call is not None and index != self.target_call):
            return output
        direction = self.vector / (self.vector.norm() + 1e-8)
        direction = direction.to(device=output.device, dtype=output.dtype)
        if self.mode == "push":
            return output + self.eps * direction
        if self.mode == "ablate":
            return output - (output @ direction).unsqueeze(-1) * direction
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
    hook = AttendHook(block.attention)
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
                print(
                    f"paired episode={episode} frame={frame} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

        clean = torch.stack([r["write_clean"] for r in records]).float()
        shifted = torch.stack([r["write_shift"] for r in records]).float()
        delta = shifted - clean
        _, s, vh = torch.linalg.svd(delta.double() - delta.double().mean(0, keepdim=True), full_matrices=False)
        k = min(args.candidates, vh.shape[0])
        candidates = vh[:k].float()  # variance-ordered candidates
        scale = float(delta.norm(dim=1).mean())
        eps = args.eps_fraction * scale
        generator = torch.Generator().manual_seed(args.seed)
        random_dirs, _ = torch.linalg.qr(
            torch.randn(clean.shape[1], args.random_controls, generator=generator)
        )
        random_dirs = random_dirs.T.float()

        attribution = {j: [] for j in range(k)}
        recovery = {j: [] for j in range(k)}
        random_recovery = {j: [] for j in range(random_dirs.shape[0])}
        random_attribution = {j: [] for j in range(random_dirs.shape[0])}
        for record in records:
            base = (record["action_shift"] - record["action_clean"]).norm().clamp_min(1e-8)
            for j in range(k):
                vector = candidates[j]
                hook.reset("push", vector=vector, eps=eps, target_call=step_index)
                pushed = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("ablate", vector=vector, target_call=step_index)
                removed = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("idle")
                attribution[j].append(float((pushed - record["action_shift"]).norm() / eps))
                recovery[j].append(float((removed - record["action_clean"]).norm() / base))
            for j in range(random_dirs.shape[0]):
                vector = random_dirs[j]
                hook.reset("push", vector=vector, eps=eps, target_call=step_index)
                pushed = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("ablate", vector=vector, target_call=step_index)
                removed = sample(model, record["images_j"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("idle")
                random_attribution[j].append(float((pushed - record["action_shift"]).norm() / eps))
                random_recovery[j].append(float((removed - record["action_clean"]).norm() / base))
            print(
                f"intervened episode={record['episode']} frame={record['frame']} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

        attribution_mean = [float(np.mean(attribution[j])) for j in range(k)]
        recovery_mean = [float(np.mean(recovery[j])) for j in range(k)]
        random_attribution_mean = [float(np.mean(random_attribution[j])) for j in range(random_dirs.shape[0])]
        random_recovery_mean = [float(np.mean(random_recovery[j])) for j in range(random_dirs.shape[0])]
        # does attribution rank the useful removals? (lower recovery = more useful)
        order_attribution = sorted(range(k), key=lambda j: attribution_mean[j], reverse=True)
        order_variance = list(range(k))  # PCA order is variance order
        top = max(1, k // 2)
        payload = {
            "schema": "dacsc_attribution_interface_probe_v1",
            "method_references": {
                "site": "cross-attention write = action-conditioning interface, arXiv:2610.20750",
                "direction_selection": "intervention attribution via small-step steering, arXiv:2610.15850",
                "directional_ablation": "Arditi et al. 2024 arXiv:2406.11717",
                "paired_render_calibration": "local lowrank_appearance_probe.py",
                "paired_flow_noise": "upstream action_generator.py:188 redraws randn; fixed per state",
            },
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "styles": sorted({r["style"] for r in records}),
                "shift": args.shift,
                "site": "action_generator.blocks[0].attention.attend (cross-attention write)",
                "euler_step": step_index,
                "states": len(records),
                "feature_dim": int(clean.shape[1]),
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "delta_spectrum": {
                "rank90": int(torch.searchsorted(torch.cumsum((s**2) / (s**2).sum(), 0), torch.tensor(0.90)).item()) + 1,
                "top1_ratio": float((s[0] ** 2) / (s**2).sum()),
                "scale": scale,
                "eps": eps,
            },
            "candidates": [
                {
                    "index": j,
                    "variance_rank": j,
                    "attribution_mean": attribution_mean[j],
                    "recovery_mean": recovery_mean[j],
                    "attribution_per_state": attribution[j],
                    "recovery_per_state": recovery[j],
                }
                for j in range(k)
            ],
            "random": [
                {
                    "index": j,
                    "attribution_mean": random_attribution_mean[j],
                    "recovery_mean": random_recovery_mean[j],
                }
                for j in range(random_dirs.shape[0])
            ],
            "selection_test": {
                "top_by_attribution": order_attribution[:top],
                "top_by_variance": order_variance[:top],
                "recovery_top_attribution": float(np.mean([recovery_mean[j] for j in order_attribution[:top]])),
                "recovery_top_variance": float(np.mean([recovery_mean[j] for j in order_variance[:top]])),
                "recovery_random": float(np.mean(random_recovery_mean)),
                "attribution_spread": float(max(attribution_mean) - min(attribution_mean)),
                "random_attribution_spread": float(max(random_attribution_mean) - min(random_attribution_mean)),
            },
            "status": "completed",
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(args.out, payload)
        print(json.dumps(payload["selection_test"], indent=2), flush=True)
        print("DONE ->", args.out, flush=True)
        return 0
    finally:
        hook.close()


if __name__ == "__main__":
    raise SystemExit(main())
