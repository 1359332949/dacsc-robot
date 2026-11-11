"""Hardening probe for the "3-dim action-relevant subspace" and "entanglement" claims.

Addresses reviewer objections raised for the V8/V9 story:

1. The V8 rank estimate used only 12 random probes, so rank90 <= 4 could be a
   probe-budget artefact. Here we use --probes (default 64) orthonormal random
   directions and report rank90/top1/stable-rank with bootstrap CIs.
2. Step dependence: the same estimate at an early Euler step.
3. Entanglement was only shown as span containment (99%). Here we compute
   principal angles between the appearance-response subspace and the
   state-response subspace, plus pairwise cosine alignment.
4. Linearity of the finite-difference response: eps sweep, response/eps ratio
   should be stable inside the linear regime.
5. Dose-response: interpolate the photometric severity (color_jitter factors
   1.6/1.6/2.2 at t=1) and check that action damage grows with severity.

Read-only, paired flow noise per state.
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
from torchvision.transforms import functional as TVF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "robocasa"))

from dacsc.corp_data import LinearNormalizer, read_episode_meta  # noqa: E402
from dacsc.corp_images import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
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
    parser.add_argument("--episodes", nargs="+", type=int, default=[16, 29, 35, 41])
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 32, 64])
    parser.add_argument("--shift", default="color_jitter")
    parser.add_argument("--probes", type=int, default=64)
    parser.add_argument("--early-probes", type=int, default=16)
    parser.add_argument("--early-step", type=int, default=4)
    parser.add_argument("--appearance-dirs", type=int, default=6)
    parser.add_argument("--state-dirs", type=int, default=3)
    parser.add_argument("--eps-factors", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--severities", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--bootstrap", type=int, default=200)
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
    if x.numel() == 0 or float(x.norm()) == 0:
        return {"degenerate": True}
    _, s, _ = torch.linalg.svd(x, full_matrices=False)
    energy = s**2
    total = float(energy.sum())
    ratio = energy / energy.sum()
    cumulative = torch.cumsum(ratio, 0)
    threshold = torch.tensor(0.90, device=cumulative.device, dtype=cumulative.dtype)
    rank90 = int(torch.searchsorted(cumulative, threshold).item()) + 1
    return {
        "rank90": rank90,
        "top1_ratio": float(ratio[0]),
        "stable_rank": float(total / float(energy[0])),
        "singular_values": [float(v) for v in s[: min(10, len(s))]],
    }


def principal_angles(basis_a: torch.Tensor, basis_b: torch.Tensor) -> list[float]:
    """Principal angles (degrees) between two orthonormal row-bases."""
    qa = torch.linalg.qr(basis_a.T.double(), mode="reduced")[0]
    qb = torch.linalg.qr(basis_b.T.double(), mode="reduced")[0]
    sv = torch.linalg.svdvals(qa.T @ qb)
    cosines = sv.clamp(-1, 1)
    return [float(math.degrees(math.acos(abs(float(c))))) for c in cosines]


def severity_view(images: dict[str, torch.Tensor], t: float) -> dict[str, torch.Tensor]:
    """Interpolate color_jitter brightness 1.6, contrast 1.6, saturation 2.2 by t."""
    brightness = 1.0 + 0.6 * t
    contrast = 1.0 + 0.6 * t
    saturation = 1.0 + 1.2 * t
    out = {}
    for name, value in images.items():
        mean = IMAGENET_MEAN.to(device=value.device, dtype=value.dtype).view(1, 3, 1, 1)
        std = IMAGENET_STD.to(device=value.device, dtype=value.dtype).view(1, 3, 1, 1)
        rgb = (value * std + mean).clamp(0, 1)
        rgb = TVF.adjust_brightness(rgb, brightness)
        rgb = TVF.adjust_contrast(rgb, contrast)
        rgb = TVF.adjust_saturation(rgb, saturation)
        out[name] = ((rgb - mean) / std).to(dtype=value.dtype)
    return out


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
        meta, states, model_xml = load_episode(args.dataset, episode)
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
                images_c, state, lang = policy_inputs(obs, info["meta"]["lang"], norm, device, 0.0, "none")
                images_j, state_j, lang_j = policy_inputs(obs, info["meta"]["lang"], norm, device, 0.0, args.shift)
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
                severity = {}
                for t in args.severities:
                    if t == 0.0:
                        severity[t] = action_c
                    else:
                        severity[t] = sample(
                            model, severity_view(images_c, t), state, lang, args.sample_steps, seed
                        )
                records.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "action_clean": action_c,
                        "action_shift": action_j,
                        "write_clean": write_c,
                        "write_shift": write_j,
                        "images_c": {k: v.clone() for k, v in images_c.items()},
                        "state": state.clone(),
                        "lang": list(lang),
                        "noise_seed": seed,
                        "severity": severity,
                    }
                )
                print(f"paired ep={episode} f={frame} elapsed={time.monotonic() - started:.1f}s", flush=True)

        dim = int(records[0]["write_clean"].numel())
        clean = torch.stack([r["write_clean"] for r in records]).float()
        shifted = torch.stack([r["write_shift"] for r in records]).float()
        delta = shifted - clean
        _, _, vh = torch.linalg.svd(delta.double() - delta.double().mean(0, keepdim=True), full_matrices=False)
        k_app = min(args.appearance_dirs, vh.shape[0])
        appearance = vh[:k_app].float()
        scale = float(delta.norm(dim=1).mean())
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        probes, _ = torch.linalg.qr(torch.randn(dim, args.probes, generator=generator))
        probes = probes.T.float()  # [P, dim]
        early_probes = probes[: args.early_probes]
        state_dirs = []
        for index in range(len(records)):
            nxt = records[(index + 1) % len(records)]["write_clean"]
            direction = (nxt - records[index]["write_clean"]).float()
            state_dirs.append(direction / direction.norm().clamp_min(1e-8))
        state_basis = torch.stack(state_dirs[: args.state_dirs])

        eps = 0.1 * scale
        rows = []
        for record in records:
            base = record["action_clean"]

            def push(vector, target_call):
                hook.reset("add", vector=vector, target_call=target_call)
                out = sample(model, record["images_c"], record["state"], record["lang"], args.sample_steps, record["noise_seed"])
                hook.reset("idle")
                return out - base

            random_responses = torch.stack([push(eps * u, last) for u in probes], dim=1)
            early_responses = torch.stack([push(eps * u, args.early_step) for u in early_probes], dim=1)
            appearance_responses = torch.stack([push(eps * u, last) for u in appearance], dim=1)
            state_responses = torch.stack([push(eps * u, last) for u in state_basis], dim=1)
            eps_sweep = {}
            for factor in args.eps_factors:
                response = push(factor * scale * appearance[0], last)
                eps_sweep[str(factor)] = {
                    "response_norm": float(response.norm()),
                    "norm_over_eps": float(response.norm() / (factor * scale)),
                }
            severity_points = {
                str(t): float((action - record["action_clean"]).norm())
                for t, action in record["severity"].items()
            }
            rows.append(
                {
                    "episode": record["episode"],
                    "frame": record["frame"],
                    "rank_full": spectrum(random_responses),
                    "rank_early": spectrum(early_responses),
                    "appearance_vs_state_angles": principal_angles(
                        appearance_responses.T, state_responses.T
                    ),
                    "appearance_span_vs_state_span_angle": principal_angles(
                        appearance_responses.T, state_responses.T
                    )[0],
                    "appearance_norm_mean": float(appearance_responses.norm(dim=0).mean()),
                    "state_norm_mean": float(state_responses.norm(dim=0).mean()),
                    "eps_sweep": eps_sweep,
                    "severity": severity_points,
                    "damage_norm": float((record["action_shift"] - record["action_clean"]).norm()),
                }
            )
            print(
                f"hard ep={record['episode']} f={record['frame']} "
                f"rank64={rows[-1]['rank_full'].get('rank90')} "
                f"rank16early={rows[-1]['rank_early'].get('rank90')} "
                f"angle={rows[-1]['appearance_span_vs_state_span_angle']:.1f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

        def bootstrap_rank(key):
            values = []
            rng = np.random.default_rng(args.seed)
            for _ in range(args.bootstrap):
                picks = rng.integers(0, len(rows), size=len(rows))
                sample_rows = [rows[i] for i in picks]
                ranks = [r[key]["rank90"] for r in sample_rows if key in r and "rank90" in r[key]]
                if ranks:
                    values.append(float(np.mean(ranks)))
            return {
                "mean": float(np.mean(values)),
                "p2_5": float(np.percentile(values, 2.5)),
                "p97_5": float(np.percentile(values, 97.5)),
            }

        severity_curve = {}
        for t in args.severities:
            severity_curve[str(t)] = float(np.mean([r["severity"][str(t)] for r in rows if str(t) in r["severity"]]))
        linearity = {}
        for factor in args.eps_factors:
            linearity[str(factor)] = float(
                np.mean([r["eps_sweep"][str(factor)]["norm_over_eps"] for r in rows])
            )

        payload = {
            "schema": "dacsc_jacobian_hardening_v1",
            "method_references": {
                "intervention_attribution": "arXiv:2610.15850",
                "non_identifiability": "arXiv:2602.06801",
                "site": "arXiv:2610.20750",
            },
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "states": len(records),
                "probes": args.probes,
                "early_probes": args.early_probes,
                "early_step": args.early_step,
                "appearance_dirs": k_app,
                "state_dirs": args.state_dirs,
                "eps": eps,
                "eps_factors": list(args.eps_factors),
                "severities": list(args.severities),
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "summary": {
                "rank_full_mean": float(np.mean([r["rank_full"]["rank90"] for r in rows])),
                "rank_full_bootstrap": bootstrap_rank("rank_full"),
                "rank_early_mean": float(np.mean([r["rank_early"]["rank90"] for r in rows if "rank90" in r["rank_early"]])),
                "rank_early_bootstrap": bootstrap_rank("rank_early"),
                "top1_ratio_mean": float(np.mean([r["rank_full"]["top1_ratio"] for r in rows])),
                "stable_rank_mean": float(np.mean([r["rank_full"]["stable_rank"] for r in rows])),
                "appearance_vs_state_angle_mean": float(np.mean([r["appearance_span_vs_state_span_angle"] for r in rows])),
                "appearance_vs_state_angle_min": float(np.min([r["appearance_span_vs_state_span_angle"] for r in rows])),
                "appearance_norm_mean": float(np.mean([r["appearance_norm_mean"] for r in rows])),
                "state_norm_mean": float(np.mean([r["state_norm_mean"] for r in rows])),
                "severity_curve": severity_curve,
                "linearity_norm_over_eps": linearity,
            },
            "states": rows,
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
