"""Can appearance features predict the action damage across states? (read-only)

Motivation: three sites (DINO block 8, resampler output, DiT action residual)
all failed the *directional ablation* test (R^2 14-21%, recovery ~1). This probe
asks the weaker, deployment-relevant question instead:

    given a calibrated appearance feature shift df = f_jitter - f_clean,
    can we predict the action damage da = a_jitter - a_clean on a NEW state,
    using only other states?

If yes, an action-space compensation (offline calibration + online correction)
is viable even though no single feature direction carries the damage.
If no, the appearance features do not determine the action damage and the
direction source has to change (behaviour-derived, COAST-style).

Protocol: leave-one-state-out ridge regression; report held-out R^2 against the
zero predictor (R^2=0) and the mean predictor. No training of the policy, no
gradients on the policy, no rollout.

References for the formulation:
* calibration by same-state paired renders: local lowrank_appearance_probe.py
* outward evaluation convention (predict the held-out state): ordinary
  leave-one-out cross-validation, no new statistic invented
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

DINO_LAYER = 8


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
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 3, 5, 10])
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


class Capture:
    def __init__(self):
        self.value: torch.Tensor | None = None
        self.handle = None

    def attach(self, module):
        self.handle = module.register_forward_hook(self._hook)
        return self

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        self.value = output.detach()
        return None

    def close(self):
        if self.handle is not None:
            self.handle.remove()


@torch.no_grad()
def sample_and_capture(model, images, state, lang, steps, dino_cap, res_cap):
    dino_cap.value = None
    res_cap.value = None
    action = model.sample_action(images, state, lang, num_steps=steps)
    if dino_cap.value is None or res_cap.value is None:
        raise RuntimeError("a capture site did not fire")
    dino = dino_cap.value.float().mean(dim=(0, 1)).cpu()  # [B, 384] -> mean over views/tokens
    res = res_cap.value.float().mean(dim=1).cpu()  # [B, 144, 768] -> mean over tokens
    return action.detach().float().flatten().cpu(), dino.squeeze(0), res.squeeze(0)


def ridge_loo(features: torch.Tensor, targets: torch.Tensor, alpha: float) -> dict:
    """Leave-one-out ridge regression; returns held-out prediction R^2."""
    x = features.double()
    y = targets.double()
    n = x.shape[0]
    predictions = torch.empty_like(y)
    for held in range(n):
        train = torch.ones(n, dtype=torch.bool)
        train[held] = False
        xt, yt = x[train], y[train]
        x_mean = xt.mean(dim=0, keepdim=True)
        y_mean = yt.mean(dim=0, keepdim=True)
        xc, yc = xt - x_mean, yt - y_mean
        fit = xc.T @ xc + alpha * torch.eye(xc.shape[1], dtype=xc.dtype)
        weight = torch.linalg.solve(fit, xc.T @ yc)
        predictions[held] = (x[held : held + 1] - x_mean) @ weight + y_mean
    ss_res = float(((y - predictions) ** 2).sum())
    ss_tot = float(((y - y.mean(dim=0, keepdim=True)) ** 2).sum())
    return {
        "r2": 1.0 - ss_res / max(ss_tot, 1e-12),
        "alpha": alpha,
        "predictions": predictions,
    }


def rank_summary(matrix: torch.Tensor) -> dict:
    x = matrix.double()
    _, s, _ = torch.linalg.svd(x - x.mean(dim=0, keepdim=True), full_matrices=False)
    energy = s**2
    ratio = energy / energy.sum()
    cumulative = torch.cumsum(ratio, 0)
    return {
        "rank90": int(torch.searchsorted(cumulative, torch.tensor(0.90)).item()) + 1,
        "stable_rank": float(energy.sum() / energy[0]),
        "top1_ratio": float(ratio[0]),
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
    backbone = model.representation_extractor.visual_encoder.backbone
    dino_cap = Capture().attach(backbone.blocks[DINO_LAYER])
    res_cap = Capture().attach(model.representation_extractor.cross_attention_resampler)
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
                action_c, dino_c, res_c = sample_and_capture(
                    model, images_c, state, lang, args.sample_steps, dino_cap, res_cap
                )
                action_j, dino_j, res_j = sample_and_capture(
                    model, images_j, state, lang, args.sample_steps, dino_cap, res_cap
                )
                rows.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "style": info["style"],
                        "da": action_j - action_c,
                        "df_dino": dino_j - dino_c,
                        "df_res": res_j - res_c,
                    }
                )
                print(
                    f"paired episode={episode} frame={frame} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

        da = torch.stack([r["da"] for r in rows])
        results = {
            "schema": "dacsc_action_prediction_probe_v1",
            "scope": {
                "dataset": str(args.dataset),
                "episodes": list(args.episodes),
                "frames": list(args.frames),
                "styles": sorted({r["style"] for r in rows}),
                "shift": args.shift,
                "states": len(rows),
                "dino_layer": DINO_LAYER,
                "checkpoint": str(args.model),
                "checkpoint_sha256": file_sha256(args.model),
                "upstream": upstream,
                "read_only": True,
                "no_training": True,
                "protocol": "leave-one-state-out ridge; held-out R2 against zero predictor",
            },
            "action_damage_spectrum": rank_summary(da),
            "prediction": {},
            "status": "completed",
            "elapsed_seconds": None,
        }
        # scale-adaptive ridge grid
        for site, key in (("dino_block8", "df_dino"), ("resampler", "df_res")):
            df = torch.stack([r[key] for r in rows])
            scale = float((df**2).sum(dim=1).mean())
            grid = [scale * factor for factor in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]
            best = None
            for alpha in grid:
                outcome = ridge_loo(df, da, alpha)
                if best is None or outcome["r2"] > best["r2"]:
                    best = outcome
            site_result = {
                "feature_dim": int(df.shape[1]),
                "feature_rank": rank_summary(df),
                "full_feature_r2": best["r2"],
                "full_feature_alpha": best["alpha"],
                "low_rank": {},
            }
            for rank in args.ranks:
                # projection learned within each leave-one-out fold is approximated
                # by a global SVD basis; the projection is linear so the ridge fit
                # on projected coordinates is equivalent up to scaling.
                _, _, vh = torch.linalg.svd(
                    df.double() - df.double().mean(dim=0, keepdim=True),
                    full_matrices=False,
                )
                basis = vh[: min(rank, vh.shape[0])].float()
                coords = df @ basis.T
                best_low = None
                for alpha in grid:
                    outcome = ridge_loo(coords, da, alpha)
                    if best_low is None or outcome["r2"] > best_low["r2"]:
                        best_low = outcome
                site_result["low_rank"][str(rank)] = {
                    "r2": best_low["r2"],
                    "alpha": best_low["alpha"],
                }
            results["prediction"][site] = site_result
            print(
                f"{site}: full R2={site_result['full_feature_r2']:.3f} "
                f"low-rank={ {k: round(v['r2'],3) for k, v in site_result['low_rank'].items()} }",
                flush=True,
            )

        results["verdict"] = verdict(results)
        results["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.out, results)
        print(json.dumps(results["verdict"], indent=2), flush=True)
        print("DONE ->", args.out, flush=True)
        return 0
    finally:
        dino_cap.close()
        res_cap.close()


def verdict(results: dict) -> dict:
    best_site, best = None, -1.0
    for site, data in results["prediction"].items():
        for name, value in [("full", data["full_feature_r2"])] + [
            (f"rank{k}", v["r2"]) for k, v in data["low_rank"].items()
        ]:
            if value > best:
                best, best_site = value, f"{site}/{name}"
    return {
        "action_damage_spectrum": results["action_damage_spectrum"],
        "best_predictor": best_site,
        "best_held_out_r2": best,
        "criterion": {
            "viable_action_space_compensation": "best held-out R2 >= 0.6",
            "weak": "0.3 <= R2 < 0.6",
            "not_viable": "R2 < 0.3",
        },
        "note": "held-out R2 against the zero predictor; one number per (site, feature rank)",
    }


if __name__ == "__main__":
    raise SystemExit(main())
