"""Closed-form low-rank action correction from calibration (read-only).

V8 showed the interface action responses live in a ~3-dim action subspace shared
by appearance and task variation. V9 therefore asks the deployment-relevant
question: can we PREDICT the action offset from signals available at test time
and subtract it?

Available at test time (no clean reference, no label):
    dz = z_shift - mu_source   (interface write deviation from the source mean)
    s  = normalised robot state
Offline (paired renders only, no policy rollouts):
    y  = a_clean - a_shift     (the action offset to cancel)
    optionally projected to the top-r action subspace of y (V8: r ~ 3)

Fitted by leave-one-state-out kernel ridge, then evaluated by
    recovery = || a_shift + y_hat - a_clean || / || a_shift - a_clean ||
Baselines: zero correction (=1), mean correction, oracle (=0).

Method family: residual action correction (ARC arXiv:2610.37547, ViTaR
arXiv:2608.15816) but fitted in closed form from paired renders instead of RL;
low-rank predictive support follows the regression-TTA literature
(arXiv:2603.09338) and the Jacobian structure measured in V8.
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
    parser.add_argument("--ranks", nargs="+", type=int, default=[0, 3, 6, 12])
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
    """Capture the cross-attention write.

    NOTE: a forward hook on the attention MODULE never fires here because
    DiTBlock.forward calls `self.attention.project_qkv(...)` and
    `self.attention.attend(...)` directly; the module itself is not called.
    Earlier probes therefore wrapped `attend` itself, and the same approach is
    used here.
    """

    def __init__(self, attention_module):
        self.attention = attention_module
        self.original = attention_module.attend
        self.value = None
        attention_module.attend = self._attend  # type: ignore[assignment]

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        self.value = output.detach().clone()
        return output

    def clear(self):
        self.value = None

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
    return action.detach().float().flatten().cpu()


def fit_predict(xtr, ytr, xte, alpha):
    k = xtr @ xtr.T
    k = k + alpha * torch.eye(k.shape[0], dtype=k.dtype)
    dual = torch.linalg.solve(k, ytr)
    return (xte @ xtr.T) @ dual


def loo_predictions(x, y, alpha):
    n = x.shape[0]
    predictions = torch.empty_like(y)
    for held in range(n):
        train = torch.ones(n, dtype=torch.bool)
        train[held] = False
        predictions[held] = fit_predict(
            x[train], y[train], x[held : held + 1], alpha
        )[0]
    return predictions


def r2(pred, target):
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean(0, keepdim=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


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
    hook = WriteHook(model.action_generator.blocks[0].attention)
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
    rows = []
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
                seed = args.seed * 1000 + len(rows)
                hook.clear()
                action_c = sample(model, images_c, state, lang, args.sample_steps, seed)
                write_c = hook.value.mean(dim=1).squeeze(0).cpu()
                hook.clear()
                action_j = sample(model, images_j, state, lang, args.sample_steps, seed)
                write_j = hook.value.mean(dim=1).squeeze(0).cpu()
                hook.clear()
                rows.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "action_clean": action_c,
                        "action_shift": action_j,
                        "write_clean": write_c,
                        "write_shift": write_j,
                        "state": state.detach().float().flatten().cpu(),
                    }
                )
                print(f"paired ep={episode} f={frame} elapsed={time.monotonic() - started:.1f}s", flush=True)

        n = len(rows)
        action_clean = torch.stack([r["action_clean"] for r in rows]).double()
        action_shift = torch.stack([r["action_shift"] for r in rows]).double()
        write_clean = torch.stack([r["write_clean"] for r in rows]).double()
        write_shift = torch.stack([r["write_shift"] for r in rows]).double()
        states = torch.stack([r["state"] for r in rows]).double()
        mu = write_clean.mean(dim=0, keepdim=True)
        dz = write_shift - mu
        damage = action_clean - action_shift

        results = {}
        for rank in args.ranks:
            if rank == 0:
                basis = torch.eye(damage.shape[1], dtype=torch.float64)
            else:
                # basis learned inside each fold below; for reporting keep global
                basis = torch.eye(damage.shape[1], dtype=torch.float64)
            for predictor_name, x in (
                ("dz", dz),
                ("dz+state", torch.cat([dz, states], dim=1)),
                ("state", states),
            ):
                key = f"rank{rank}|{predictor_name}"
                predictions = torch.empty_like(damage)
                for held in range(n):
                    train = torch.ones(n, dtype=torch.bool)
                    train[held] = False
                    if rank == 0:
                        fold_basis = torch.eye(damage.shape[1], dtype=torch.float64)
                    else:
                        _, _, vh = torch.linalg.svd(damage[train], full_matrices=False)
                        r = min(rank, vh.shape[0])
                        fold_basis = vh[:r].T  # [120, r]
                    y_train = damage[train] @ fold_basis
                    xtr = x[train]
                    x_mean = xtr.mean(dim=0, keepdim=True)
                    y_mean = y_train.mean(dim=0, keepdim=True)
                    scale = float((xtr - x_mean).pow(2).sum(dim=1).mean())
                    best = None
                    for factor in (1e-4, 1e-3, 1e-2, 1e-1):
                        alpha = max(factor * scale, 1e-8)
                        # inner leave-one-out over training rows
                        inner = torch.empty_like(y_train)
                        m = xtr.shape[0]
                        for inner_held in range(m):
                            mask = torch.ones(m, dtype=torch.bool)
                            mask[inner_held] = False
                            pred = fit_predict(
                                xtr[mask] - x_mean,
                                y_train[mask] - y_mean,
                                xtr[inner_held : inner_held + 1] - x_mean,
                                alpha,
                            )
                            inner[inner_held] = pred[0] + y_mean[0]
                        score = r2(inner, y_train)
                        if best is None or score > best[0]:
                            best = (score, alpha)
                    alpha = best[1]
                    pred_low = fit_predict(
                        xtr - x_mean, y_train - y_mean, x[held : held + 1] - x_mean, alpha
                    )[0] + y_mean[0]
                    predictions[held] = fold_basis @ pred_low
                squared = (predictions - damage) ** 2
                total = ((damage - damage.mean(0, keepdim=True)) ** 2)
                results[key] = {
                    "r2": 1.0 - float(squared.sum() / total.sum().clamp_min(1e-12)),
                    "recovery_mean": float(
                        ((action_shift + predictions - action_clean) ** 2).sum(dim=1).sqrt()
                        .div(((action_shift - action_clean) ** 2).sum(dim=1).sqrt().clamp_min(1e-12))
                        .mean()
                    ),
                    "recovery_median": float(
                        ((action_shift + predictions - action_clean) ** 2).sum(dim=1).sqrt()
                        .div(((action_shift - action_clean) ** 2).sum(dim=1).sqrt().clamp_min(1e-12))
                        .median()
                    ),
                }
                print(f"{key}: R2={results[key]['r2']:.3f} recovery_med={results[key]['recovery_median']:.3f}", flush=True)

        # baselines
        mean_pred = damage.mean(dim=0, keepdim=True).expand_as(damage)
        baselines = {
            "zero": {
                "recovery_mean": 1.0,
                "recovery_median": 1.0,
            },
            "mean_correction": {
                "recovery_mean": float(
                    ((action_shift + mean_pred - action_clean) ** 2).sum(dim=1).sqrt()
                    .div(((action_shift - action_clean) ** 2).sum(dim=1).sqrt().clamp_min(1e-12))
                    .mean()
                ),
                "recovery_median": float(
                    ((action_shift + mean_pred - action_clean) ** 2).sum(dim=1).sqrt()
                    .div(((action_shift - action_clean) ** 2).sum(dim=1).sqrt().clamp_min(1e-12))
                    .median()
                ),
            },
            "oracle": {"recovery_mean": 0.0, "recovery_median": 0.0},
        }

        payload = {
            "schema": "dacsc_lowrank_action_correction_v1",
            "method_references": {
                "residual_action_correction": "ARC arXiv:2610.37547; ViTaR arXiv:2608.15816",
                "low_rank_predictive_support": "regression TTA arXiv:2603.09338",
                "jacobian_subspace": "V8 of this project; site arXiv:2610.20750",
                "paired_flow_noise": "upstream action_generator.py:188",
            },
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "shift": args.shift,
                "states": n,
                "predictors": ["dz", "dz+state", "state"],
                "ranks": list(args.ranks),
                "protocol": "leave-one-state-out; inner leave-one-out alpha selection",
                "uses_clean_action_at_test": False,
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
            },
            "results": results,
            "baselines": baselines,
            "status": "completed",
            "elapsed_seconds": None,
        }
        payload["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.out, payload)
        print(json.dumps({"baselines": baselines}, indent=2), flush=True)
        print("DONE ->", args.out, flush=True)
        return 0
    finally:
        hook.close()


if __name__ == "__main__":
    raise SystemExit(main())
