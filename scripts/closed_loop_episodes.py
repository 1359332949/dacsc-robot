"""Closed-loop success rate starting from recorded dataset episode states.

The manifest-based evaluator (eval_corp_robocasa_sr.py) creates scenes and rolls
out from the scene initial state, which produced 0/3 successes even in the clean
condition on layout 58. Earlier ad-hoc runs that started from the RECORDED
episode state (model.xml + states[0]) did succeed on layout 31, so this script
reproduces that setup:

  env = RoboCasaGymEnv(task, style_ids=[style], layout_ids=[layout])
  reset_to(core, {model: episode model.xml, states: episode states[0], ep_meta})
  rollout_episode(...)   # the project's own rollout loop, imported, not copied

Optional PSC: when --psc-stats is given, the source-support projection is
installed around every model call (dacsc/psc.py).
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
from dacsc.psc import install, load_stats  # noqa: E402
from dacsc.rollout_protocol import rollout_episode  # noqa: E402
from dacsc.upstream import verify_upstream  # noqa: E402
from scripts.eval_corp_robocasa_sr import policy_inputs  # noqa: E402
from robocasa.utils.env_utils import convert_action  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", nargs="+", type=int, required=True)
    ap.add_argument("--shift", default="color_jitter")
    ap.add_argument("--blur-sigma", type=float, default=0.0)
    ap.add_argument("--action-steps", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=750)
    ap.add_argument("--psc-stats", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    upstream = verify_upstream()
    from robocasa.models.scenes.scene_registry import LayoutType
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    metadata = read_episode_meta(args.dataset)
    device = torch.device(args.device)
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    model.eval()
    norm = LinearNormalizer(mean, std)
    stats = load_stats(args.psc_stats) if args.psc_stats is not None else None
    records = []
    for episode in args.episodes:
        extra = args.dataset / "extras" / f"episode_{episode:06d}"
        meta = json.loads((extra / "ep_meta.json").read_text())
        with np.load(extra / "states.npz") as f:
            states = np.asarray(f["states"])
        with gzip.open(extra / "model.xml.gz", "rb") as f:
            model_xml = f.read().decode()
        env = RoboCasaGymEnv(
            env_name=args.task,
            split=None,
            obj_instance_split="pretrain",
            style_ids=[meta["style_id"]],
            layout_ids=[meta["layout_id"]],
            camera_widths=256,
            camera_heights=256,
            seed=args.seed,
        )
        core = env.env
        projector = None
        try:
            reset_to(core, {"model": model_xml, "states": states[0], "ep_meta": json.dumps(meta)})
            obs = env.get_observation(core._get_observations(force_update=True))
            if stats is not None:
                projector = install(model, stats)
            result = rollout_episode(
                env,
                model,
                norm,
                device,
                obs,
                meta["lang"],
                policy_inputs=policy_inputs,
                convert_action=convert_action,
                max_steps=args.max_steps,
                action_steps=args.action_steps,
                blur_sigma=args.blur_sigma,
                shift=args.shift,
            )
        finally:
            if projector is not None:
                projector.close()
            env.close()
        records.append({"episode": episode, "style": meta["style_id"], "layout": meta["layout_id"], **result})
        print(f"[{args.task}] ep={episode} style={meta['style_id']} layout={meta['layout_id']} "
              f"shift={args.shift} success={int(result['success'])} steps={result['steps']}", flush=True)

    sr = sum(r["success"] for r in records) / len(records)
    payload = {
        "schema": "dacsc_closed_loop_episodes_v1",
        "task": args.task,
        "shift": args.shift,
        "blur_sigma": args.blur_sigma,
        "psc_stats": str(args.psc_stats) if args.psc_stats else None,
        "action_steps": args.action_steps,
        "max_steps": args.max_steps,
        "episodes": records,
        "successes": sum(r["success"] for r in records),
        "success_rate": sr,
        "checkpoint": args.model,
        "upstream": upstream,
        "read_only": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"task": args.task, "shift": args.shift, "psc": bool(stats), "sr": sr}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
