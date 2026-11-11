"""Build PSC source statistics (mean + top-r principal subspace) for one task.

The statistics come from clean renders of CALIBRATION episodes only (val styles
51-60 by default), replayed in the simulator; no test-time information and no
rollouts are used. Saved as JSON so the closed-loop evaluator can load them.
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


class Capture:
    def __init__(self, attention_module):
        self.attention = attention_module
        self.original = attention_module.attend
        self.value = None
        attention_module.attend = self._capture  # type: ignore[assignment]

    def _capture(self, *args, **kwargs):
        output = self.original(*args, **kwargs)
        self.value = output.detach().clone()
        return output

    def close(self):
        self.attention.attend = self.original  # type: ignore[assignment]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 32, 64, 96, 128])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--style-min", type=int, default=51)
    ap.add_argument("--style-max", type=int, default=60)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    upstream = verify_upstream()
    from robocasa.models.scenes.scene_registry import LayoutType
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    metadata = read_episode_meta(args.dataset)
    val = [e for e, m in metadata.items() if args.style_min <= int(m["style_id"]) <= args.style_max][: args.episodes]
    if not val:
        raise ValueError("no val-style calibration episodes")
    episodes = {}
    for episode in val:
        extra = args.dataset / "extras" / f"episode_{episode:06d}"
        meta = json.loads((extra / "ep_meta.json").read_text())
        with np.load(extra / "states.npz") as state_file:
            states = np.asarray(state_file["states"])
        with gzip.open(extra / "model.xml.gz", "rb") as model_file:
            model_xml = model_file.read().decode()
        episodes[episode] = {"meta": meta, "states": states, "model_xml": model_xml}

    device = torch.device(args.device)
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    model.eval()
    norm = LinearNormalizer(mean, std)
    capture = Capture(model.action_generator.blocks[0].attention)
    styles = sorted({info["meta"]["style_id"] for info in episodes.values()})
    env = RoboCasaGymEnv(
        env_name=args.task,
        split=None,
        obj_instance_split="pretrain",
        style_ids=styles,
        layout_ids=LayoutType.TRAIN,
        camera_widths=256,
        camera_heights=256,
        seed=args.seed,
    )
    core = env.env
    features = []
    try:
        for episode, info in episodes.items():
            for frame in args.frames:
                if frame >= info["states"].shape[0]:
                    continue
                reset_to(core, {"model": info["model_xml"], "states": info["states"][frame], "ep_meta": json.dumps(info["meta"])})
                obs = env.get_observation(core._get_observations(force_update=True))
                images, state, lang = policy_inputs(obs, info["meta"]["lang"], norm, device, 0.0, "none")
                capture.value = None
                with torch.no_grad():
                    model.sample_action(images, state, lang)
                features.append(capture.value.mean(dim=1).squeeze(0).cpu())
    finally:
        capture.close()
        env.close()
    matrix = torch.stack(features).float()
    mu = matrix.mean(dim=0)
    _, s, vh = torch.linalg.svd(matrix - mu, full_matrices=False)
    rank = min(args.rank, vh.shape[0])
    basis = vh[:rank].float()
    centred = matrix - mu
    coords = centred @ basis.T
    sigma_s = coords.std(dim=0)
    residual = centred - coords @ basis
    tau = float(residual.pow(2).mean())
    payload = {
        "schema": "dacsc_psc_source_stats_v1",
        "task": args.task,
        "dataset": str(args.dataset),
        "episodes": list(episodes),
        "frames": args.frames,
        "rank": rank,
        "n_features": int(matrix.shape[0]),
        "mean": mu.tolist(),
        "basis": basis.tolist(),
        "singular_values": [float(v) for v in s[: rank]],
        "sigma_s": sigma_s.tolist(),
        "tau": tau,
        "checkpoint": args.model,
        "checkpoint_sha256": file_sha256(Path(args.model)),
        "upstream": upstream,
        "read_only": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    print(json.dumps({"task": args.task, "rank": rank, "n": int(matrix.shape[0])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
