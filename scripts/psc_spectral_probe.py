"""Spectral recalibration of PSC (source-free, uses only unlabeled test batch).

PSC (arXiv:2603.09338) does not merely project; it matches SUPPORT spectra and
RESIDUAL spectra between source and target, using mini-batch statistics of the
unlabeled test stream:
  residual model: source r ~ N(0, tau I_{D-K}); target r_hat ~ N(mu_perp, nu_perp I)
  support model : source coordinates have per-direction scale sigma_s

Our earlier probe implemented only the hard projection z' = mu + P(z-mu), i.e.
support projection with unit rescaling and no residual calibration. This probe
implements the spectral version:

  centred = z - mu_hat_t                     (target batch mean)
  c = centred @ U_K.T                        (support coordinates)
  r = centred - c @ U_K                      (residual coordinates)
  c' = c * (sigma_s / sigma_hat_t)           (support spectral matching)
  r' = r * (sqrt(tau) / sqrt(nu_hat))        (residual spectral matching)
  z' = mu_s + c' @ U_K + r'

All target statistics (mu_hat_t, sigma_hat_t, nu_hat) are estimated from the
unlabeled test window only; the source statistics come from source-style clean
renders offline. No clean target render, no label, no gradient.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
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


class SpectralHook:
    def __init__(self, attention_module):
        self.attention = attention_module
        self.original = attention_module.attend
        self.mode = "idle"
        self.params = None
        self.call = 0
        self.value = None
        attention_module.attend = self._attend  # type: ignore[assignment]

    def reset(self, mode="idle", params=None):
        self.mode = mode
        self.params = params
        self.call = 0
        self.value = None

    def _attend(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        self.call += 1
        if self.mode == "capture":
            self.value = output.detach().clone()
            return output
        if self.mode == "spectral" and self.params is not None:
            p = self.params
            mu_s = p["mu_s"].to(output.device, output.dtype)
            mu_t = p["mu_t"].to(output.device, output.dtype)
            basis = p["basis"].to(output.device, output.dtype)
            scale_c = p["scale_c"].to(output.device, output.dtype)
            scale_r = p["scale_r"].to(output.device, output.dtype)
            centred = output - mu_t
            c = centred @ basis.T
            r = centred - c @ basis
            return mu_s + (c * scale_c) @ basis + r * scale_r
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 32, 64])
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--shift", default="color_jitter")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    upstream = verify_upstream()
    from robocasa.models.scenes.scene_registry import LayoutType
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    stats = json.loads(args.stats.read_text())
    mu_s = torch.tensor(stats["mean"], dtype=torch.float32)
    basis = torch.tensor(stats["basis"], dtype=torch.float32)
    sigma_s = torch.tensor(stats["sigma_s"], dtype=torch.float32)
    tau = float(stats["tau"])

    metadata = read_episode_meta(args.dataset)
    episodes = [e for e, m in metadata.items() if 51 <= int(m["style_id"]) <= 60][: args.episodes]
    data = {}
    for episode in episodes:
        extra = args.dataset / "extras" / f"episode_{episode:06d}"
        meta = json.loads((extra / "ep_meta.json").read_text())
        with np.load(extra / "states.npz") as f:
            states = np.asarray(f["states"])
        with gzip.open(extra / "model.xml.gz", "rb") as f:
            model_xml = f.read().decode()
        data[episode] = {"meta": meta, "states": states, "model_xml": model_xml}

    device = torch.device(args.device)
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    model.eval()
    norm = LinearNormalizer(mean, std)
    hook = SpectralHook(model.action_generator.blocks[0].attention)
    steps = model.action_generator.num_sample_steps
    styles = sorted({info["meta"]["style_id"] for info in data.values()})
    env = RoboCasaGymEnv(env_name=args.task, split=None, obj_instance_split="pretrain",
                         style_ids=styles, layout_ids=LayoutType.TRAIN,
                         camera_widths=256, camera_heights=256, seed=args.seed)
    core = env.env
    rows = []
    try:
        for episode, info in data.items():
            for frame in args.frames:
                if frame >= info["states"].shape[0]:
                    continue
                reset_to(core, {"model": info["model_xml"], "states": info["states"][frame],
                                "ep_meta": json.dumps(info["meta"])})
                obs = env.get_observation(core._get_observations(force_update=True))
                images_c, state, lang = policy_inputs(obs, info["meta"]["lang"], norm, device, 0.0, "none")
                images_j, _, _ = policy_inputs(obs, info["meta"]["lang"], norm, device, 0.0, args.shift)
                seed = args.seed * 1000 + len(rows)
                hook.reset("capture")
                a_c = sample(model, images_c, state, lang, None, seed)
                write_c = hook.value.mean(dim=1).squeeze(0).cpu()
                hook.reset("capture")
                a_j = sample(model, images_j, state, lang, None, seed)
                write_j = hook.value.mean(dim=1).squeeze(0).cpu()
                hook.reset("idle")
                rows.append({"episode": episode, "frame": frame, "a_c": a_c, "a_j": a_j,
                             "write_c": write_c, "write_j": write_j,
                             "images_j": {k: v.clone() for k, v in images_j.items()},
                             "state": state.clone(), "lang": list(lang), "seed": seed})
    finally:
        env.close()

    # hard projection baseline (PSC support projection, source mu/basis)
    results = {"no_adapt": [], "hard_projection": [], "spectral": [], "spectral_support_only": [], "spectral_residual_only": []}
    test_writes = torch.stack([r["write_j"] for r in rows]).float()
    for record in rows:
        base = (record["a_j"] - record["a_c"]).norm().clamp_min(1e-8)
        results["no_adapt"].append(1.0)

        hook.reset("spectral", params={"mu_s": mu_s, "mu_t": mu_s, "basis": basis,
                                       "scale_c": torch.ones(basis.shape[0]), "scale_r": torch.ones(1)})
        adapted = sample(model, record["images_j"], record["state"], record["lang"], None, record["seed"])
        hook.reset("idle")
        results["hard_projection"].append(float((adapted - record["a_c"]).norm() / base))

    # windows of the unlabeled test stream
    window = args.window
    for start in range(0, len(rows), window):
        chunk = rows[start:start + window]
        z = test_writes[start:start + window]
        mu_t = z.mean(dim=0)
        centred = z - mu_t
        c = centred @ basis.T
        sigma_t = c.std(dim=0).clamp_min(1e-6)
        residual = centred - c @ basis
        nu = float(residual.pow(2).mean().clamp_min(1e-8))
        scale_c = (sigma_s / sigma_t).clamp(0.5, 2.0)
        scale_r = torch.tensor([min(max((tau / nu) ** 0.5, 0.5), 2.0)])
        variants = {
            "spectral": (scale_c, scale_r),
            "spectral_support_only": (scale_c, torch.ones(1)),
            "spectral_residual_only": (torch.ones(basis.shape[0]), scale_r),
        }
        for name, (sc, sr) in variants.items():
            hook.reset("spectral", params={"mu_s": mu_s, "mu_t": mu_t, "basis": basis,
                                           "scale_c": sc, "scale_r": sr})
            for record in chunk:
                adapted = sample(model, record["images_j"], record["state"], record["lang"], None, record["seed"])
                base = (record["a_j"] - record["a_c"]).norm().clamp_min(1e-8)
                results[name].append(float((adapted - record["a_c"]).norm() / base))
            hook.reset("idle")

    def summarize(values):
        return {"mean": float(np.mean(values)), "median": float(np.median(values)),
                "below_0.8": int(sum(1 for v in values if v < 0.8)),
                "above_1.5": int(sum(1 for v in values if v > 1.5))}

    payload = {
        "schema": "dacsc_psc_spectral_v1",
        "task": args.task,
        "states": len(rows),
        "window": window,
        "rank": int(basis.shape[0]),
        "summary": {k: summarize(v) for k, v in results.items()},
        "checkpoint_sha256": file_sha256(args.model),
        "upstream": upstream,
        "source_free": True,
        "read_only": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
