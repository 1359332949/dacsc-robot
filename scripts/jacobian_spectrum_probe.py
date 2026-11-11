"""Action-Jacobian spectrum at the cross-attention interface (read-only).

Question: does the policy have a structured "action-relevant subspace" at the
interface, and do the appearance directions live inside it or outside it?

For each state we push a set of unit directions into the cross-attention write
at the last Euler step (finite differences) and record the 120-dim action
response:
    random      : 12 orthonormal random directions (probe J's image)
    appearance  : top-6 PCA directions of the same-state paired appearance delta
    state       : 3 differences between consecutive states' clean writes
Then we report
  * the singular spectrum of the random-probe response matrix -> is J's image low-rank?
  * ||J u|| for appearance vs random vs state directions -> are appearance
    directions "quiet" or "loud"?
  * how much of each appearance response is spanned by the random-probe image
    -> are appearance directions action-relevant?

Grounding: intervention attribution / small-step steering (arXiv:2610.15850),
non-identifiability of single directions (arXiv:2602.06801), source predictive
support for regression TTA (arXiv:2603.09338), action-conditioning interface
(arXiv:2610.20750). Paired flow-matching noise fixed per state.
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
    parser.add_argument("--random-probes", type=int, default=12)
    parser.add_argument("--appearance-dirs", type=int, default=6)
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


def spectrum(matrix: torch.Tensor) -> dict:
    x = matrix.double()
    if x.numel() == 0:
        return {"degenerate": True}
    _, s, _ = torch.linalg.svd(x, full_matrices=False)
    energy = s**2
    total = float(energy.sum())
    if total <= 0:
        return {"degenerate": True}
    ratio = energy / energy.sum()
    cumulative = torch.cumsum(ratio, 0)
    threshold = torch.tensor(0.90, device=cumulative.device, dtype=cumulative.dtype)
    rank90 = int(torch.searchsorted(cumulative, threshold).item()) + 1
    return {
        "rank90": rank90,
        "top1_ratio": float(ratio[0]),
        "top3_ratio": float(cumulative[min(2, len(cumulative) - 1)]),
        "stable_rank": float(total / energy[0]),
        "singular_values": [float(v) for v in s[: min(12, len(s))]],
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
    block = model.action_generator.blocks[0]
    hook = WriteHook(block.attention)
    steps = args.sample_steps or model.action_generator.num_sample_steps
    last = steps - 1
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

        dim = int(records[0]["write_clean"].numel())
        clean = torch.stack([r["write_clean"] for r in records]).float()
        shifted = torch.stack([r["write_shift"] for r in records]).float()
        delta = shifted - clean
        _, _, vh = torch.linalg.svd(
            delta.double() - delta.double().mean(0, keepdim=True), full_matrices=False
        )
        k_app = min(args.appearance_dirs, vh.shape[0])
        appearance = vh[:k_app].float()
        scale = float(delta.norm(dim=1).mean())
        eps = args.eps_fraction * scale
        generator = torch.Generator().manual_seed(args.seed)
        random_dirs, _ = torch.linalg.qr(
            torch.randn(dim, args.random_probes, generator=generator)
        )
        random_dirs = random_dirs.T.float()

        per_state = []
        for index, record in enumerate(records):
            base = record["action_clean"]
            responses = {}
            for name, dirs in (("random", random_dirs), ("appearance", appearance)):
                vectors = []
                for j in range(dirs.shape[0]):
                    hook.reset("add", vector=eps * dirs[j], target_call=last)
                    pushed = sample(model, record["images_c"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                    hook.reset("idle")
                    vectors.append(pushed - base)
                responses[name] = torch.stack(vectors, dim=1)  # [120, p]
            # state directions: clean write difference to the next state
            next_clean = records[(index + 1) % len(records)]["write_clean"]
            state_dir = (next_clean - record["write_clean"]).float()
            state_dir = state_dir / state_dir.norm().clamp_min(1e-8)
            hook.reset("add", vector=eps * state_dir, target_call=last)
            pushed = sample(model, record["images_c"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
            hook.reset("idle")
            state_response = (pushed - base).unsqueeze(1)

            random_norms = responses["random"].norm(dim=0)
            appearance_norms = responses["appearance"].norm(dim=0)
            # how much of each appearance response lies in the random-probe image
            basis = responses["random"].double()
            q, _ = torch.linalg.qr(basis)
            appearance_d = responses["appearance"].double()
            projection = q @ (q.T @ appearance_d)
            captured = (projection**2).sum(dim=0) / (appearance_d**2).sum(dim=0).clamp_min(1e-12)
            per_state.append(
                {
                    "episode": record["episode"],
                    "frame": record["frame"],
                    "random_response_spectrum": spectrum(responses["random"]),
                    "random_norm_mean": float(random_norms.mean()),
                    "appearance_norm_mean": float(appearance_norms.mean()),
                    "appearance_norm_per_dir": [float(v) for v in appearance_norms],
                    "appearance_over_random": float(appearance_norms.mean() / random_norms.mean().clamp_min(1e-12)),
                    "state_norm": float(state_response.norm()),
                    "state_over_random": float(state_response.norm() / random_norms.mean().clamp_min(1e-12)),
                    "appearance_captured_by_random_span": [float(v) for v in captured],
                }
            )
            print(
                f"jac ep={record['episode']} f={record['frame']} "
                f"app/rand={per_state[-1]['appearance_over_random']:.3f} "
                f"state/rand={per_state[-1]['state_over_random']:.3f} "
                f"captured={np.mean(per_state[-1]['appearance_captured_by_random_span']):.3f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

        def agg(key):
            values = [s[key] for s in per_state]
            return {"mean": float(np.mean(values)), "median": float(np.median(values))}

        payload = {
            "schema": "dacsc_action_jacobian_spectrum_v1",
            "method_references": {
                "intervention_attribution": "arXiv:2610.15850",
                "non_identifiability": "arXiv:2602.06801",
                "source_predictive_support": "arXiv:2603.09338",
                "site": "arXiv:2610.20750",
                "paired_flow_noise": "upstream action_generator.py:188",
            },
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "styles": sorted({r["style"] for r in records}),
                "shift": args.shift,
                "site": "action_generator.blocks[0].attention.attend",
                "euler_step": last,
                "states": len(records),
                "feature_dim": dim,
                "action_dim": int(records[0]["action_clean"].numel()),
                "random_probes": args.random_probes,
                "appearance_dirs": k_app,
                "eps": eps,
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "summary": {
                "appearance_over_random": agg("appearance_over_random"),
                "state_over_random": agg("state_over_random"),
                "random_response_rank90": [s["random_response_spectrum"] for s in per_state],
                "appearance_captured_by_random_span_mean": float(
                    np.mean([np.mean(s["appearance_captured_by_random_span"]) for s in per_state])
                ),
                "appearance_captured_by_random_span_median": float(
                    np.median([np.mean(s["appearance_captured_by_random_span"]) for s in per_state])
                ),
            },
            "states": per_state,
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
