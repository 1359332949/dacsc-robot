"""Explain WHY PSC helps some tasks more than others.

Per task, per state:
  damage                ||a_shift - a_clean||
  write_delta_ratio     ||z_shift - z_clean|| / ||z_clean||
  residual_fraction     ||(I-P)(z_shift - mu)|| / ||z_shift - mu||
                        (how much of the shifted write lies OUTSIDE the source
                         support; PSC drops exactly this part)
  psc_recovery          ||a_proj - a_clean|| / ||a_shift - a_clean||
and the correlations across states between psc_recovery and the two shift
magnitudes. This tells us whether task differences come from a weaker shift or
from the shift lying inside the source support.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
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


class WriteHook:
    def __init__(self, attention_module):
        self.attention = attention_module
        self.original = attention_module.attend
        self.mode = "idle"
        self.mean = None
        self.basis = None
        self.target_calls = None
        self.call = 0
        self.value = None
        attention_module.attend = self._attend  # type: ignore[assignment]

    def reset(self, mode="idle", mean=None, basis=None, target_calls=None):
        self.mode = mode
        self.mean = mean
        self.basis = basis
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
            return mean + (centred @ basis.T) @ basis
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


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 32, 64])
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
    hook = WriteHook(model.action_generator.blocks[0].attention)
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

    clean = torch.stack([r["write_c"] for r in rows]).float()
    for index, record in enumerate(rows):
        train = torch.ones(len(rows), dtype=torch.bool)
        train[index] = False
        source = clean[train]
        mu = source.mean(dim=0)
        _, _, vh = torch.linalg.svd(source - mu, full_matrices=False)
        r = min(args.rank, vh.shape[0])
        basis = vh[:r].float()
        delta = (record["write_j"] - mu).float()
        inside = (delta @ basis.T) @ basis
        outside = delta - inside
        record["residual_fraction"] = float(outside.norm() / delta.norm().clamp_min(1e-8))
        record["write_delta_ratio"] = float((record["write_j"] - record["write_c"]).norm() / record["write_c"].norm().clamp_min(1e-8))
        hook.reset("project", mean=mu, basis=basis, target_calls=set(range(steps)))
        adapted = sample(model, record["images_j"], record["state"], record["lang"], None, record["seed"])
        hook.reset("idle")
        record["psc_recovery"] = float((adapted - record["a_c"]).norm() / (record["a_j"] - record["a_c"]).norm().clamp_min(1e-8))
        record["damage"] = float((record["a_j"] - record["a_c"]).norm())

    payload = {
        "schema": "dacsc_psc_diagnostics_v1",
        "task": args.task,
        "states": len(rows),
        "rank": args.rank,
        "summary": {
            "damage_mean": float(np.mean([r["damage"] for r in rows])),
            "write_delta_ratio_mean": float(np.mean([r["write_delta_ratio"] for r in rows])),
            "residual_fraction_mean": float(np.mean([r["residual_fraction"] for r in rows])),
            "psc_recovery_mean": float(np.mean([r["psc_recovery"] for r in rows])),
            "psc_recovery_median": float(np.median([r["psc_recovery"] for r in rows])),
            "corr_recovery_residual_fraction": correlation(
                [r["psc_recovery"] for r in rows], [r["residual_fraction"] for r in rows]),
            "corr_recovery_damage": correlation(
                [r["psc_recovery"] for r in rows], [r["damage"] for r in rows]),
        },
        "checkpoint_sha256": file_sha256(args.model),
        "upstream": upstream,
        "read_only": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    # drop tensors from state list for JSON
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
