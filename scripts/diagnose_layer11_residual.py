"""Test whether a clean layer11 residual explains low-action-error drift.

Condition:
    y_mixed = x_clean + a_jitter + f_jitter

This keeps jitter attention and FFN writes while replacing only the block input
residual. It is a privileged diagnostic, not a deployable intervention.
"""

from __future__ import annotations

import argparse
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
from scripts.diagnose_layer11_accumulation import (  # noqa: E402
    ACTION_STEPS,
    MAX_STEPS,
    action_metrics,
    atomic_json,
    drawer_open,
    eef_position,
    gripper_qpos,
    load_episode,
    sample_action,
    summarize,
)
from scripts.eval_corp_robocasa_sr import policy_inputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actions-out", type=Path, required=True)
    parser.add_argument("--episodes", default="10,50")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def capture_clean_x(block, visual, images) -> torch.Tensor:
    captured = {}
    hook = block.register_forward_pre_hook(
        lambda _module, inputs: captured.__setitem__(
            "x", inputs[0].detach().clone()
        )
    )
    try:
        with torch.inference_mode():
            visual(images)
    finally:
        hook.remove()
    if "x" not in captured:
        raise RuntimeError("failed to capture clean layer11 residual input")
    return captured["x"]


def install_residual_hooks(block, clean_x: torch.Tensor):
    captured = {}
    hooks = [
        block.ls1.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__(
                "attention", output.detach().clone()
            )
        ),
        block.ls2.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__(
                "ffn", output.detach().clone()
            )
        ),
    ]

    def replace_output(_module, _inputs, output):
        if set(captured) != {"attention", "ffn"}:
            raise RuntimeError("jitter branch capture is incomplete")
        return (
            clean_x.to(device=output.device, dtype=output.dtype)
            + captured["attention"].to(device=output.device, dtype=output.dtype)
            + captured["ffn"].to(device=output.device, dtype=output.dtype)
        )

    hooks.append(block.register_forward_hook(replace_output))
    return hooks


def main() -> int:
    args = parse_args()
    if args.out.exists() or args.actions_out.exists():
        raise FileExistsError("refusing to overwrite residual diagnostic outputs")
    episodes = tuple(int(item) for item in args.episodes.split(","))
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError("episodes must be a non-empty unique list")
    upstream = verify_upstream()
    from robocasa.models.scenes.scene_registry import LayoutType
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.utils.env_utils import convert_action
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    metadata = read_episode_meta(args.dataset)
    episode_data = {}
    for episode in episodes:
        if episode not in metadata:
            raise ValueError(f"episode is absent: {episode}")
        meta, initial_state, model_xml = load_episode(args.dataset, episode)
        refs = meta.get("fixture_refs")
        if not isinstance(refs, dict) or not refs.get("drawer"):
            raise ValueError(f"episode {episode} lacks a drawer fixture ref")
        episode_data[episode] = {
            "style": metadata[episode]["style_id"],
            "layout": metadata[episode]["layout_id"],
            "target_drawer": refs["drawer"],
            "meta": meta,
            "initial_state": initial_state,
            "model_xml": model_xml,
        }

    checkpoint_sha256 = file_sha256(args.model)
    device = torch.device(args.device)
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    norm = LinearNormalizer(mean, std)
    visual = model.representation_extractor.visual_encoder
    block = visual.backbone.blocks[11]
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
    started = time.monotonic()
    payload = {
        "schema": "dacsc_layer11_residual_diagnostic_v1",
        "started_at_epoch": started,
        "scope": {
            "episodes": list(episodes),
            "condition": "x_clean + a_jitter + f_jitter",
            "initial_state": "official training episode frame0",
            "action_steps": ACTION_STEPS,
            "max_steps": MAX_STEPS,
            "policy_seed_formula": "100000 + episode * 1000 + chunk",
            "checkpoint": str(args.model),
            "checkpoint_sha256": checkpoint_sha256,
            "upstream": upstream,
            "clean_reference_is_privileged": True,
            "deployable_method": False,
        },
        "results": [],
        "status": "running",
    }
    atomic_json(args.out, payload)
    action_arrays = {}
    try:
        for episode in episodes:
            info = episode_data[episode]
            reset_to(
                core,
                {
                    "model": info["model_xml"],
                    "states": info["initial_state"],
                    "ep_meta": json.dumps(info["meta"]),
                },
            )
            obs = env.get_observation(core._get_observations(force_update=True))
            success = False
            done = False
            truncated = False
            steps = 0
            chunks = []
            actions = []
            while not (success or done or truncated) and steps < MAX_STEPS:
                clean_images, state, lang = policy_inputs(
                    obs, info["meta"]["lang"], norm, device, 0.0, "none"
                )
                jitter_images, jitter_state, jitter_lang = policy_inputs(
                    obs, info["meta"]["lang"], norm, device, 0.0, "color_jitter"
                )
                if not torch.equal(state, jitter_state) or lang != jitter_lang:
                    raise RuntimeError("clean and jitter policy inputs are not paired")
                policy_seed = 100000 + episode * 1000 + len(chunks)
                clean_action = sample_action(
                    model, clean_images, state, lang, policy_seed
                )
                clean_x = capture_clean_x(
                    block,
                    visual,
                    torch.cat([clean_images[view] for view in model.view_names]),
                )
                hooks = install_residual_hooks(block, clean_x)
                try:
                    action = sample_action(
                        model,
                        jitter_images,
                        jitter_state,
                        jitter_lang,
                        policy_seed,
                    )
                finally:
                    for hook in hooks:
                        hook.remove()
                row = {
                    "chunk": len(chunks),
                    "step_start": steps,
                    **action_metrics(action, clean_action),
                    "drawer_start": drawer_open(core),
                    "eef_start": eef_position(obs),
                    "gripper_qpos_start": gripper_qpos(obs),
                }
                for raw_action in action[:ACTION_STEPS]:
                    obs, _, done, truncated, info_step = env.step(
                        convert_action(
                            np.clip(raw_action, -1, 1).astype(np.float32)
                        )
                    )
                    steps += 1
                    success = bool(info_step["success"])
                    if success or done or truncated or steps >= MAX_STEPS:
                        break
                row.update(
                    {
                        "step_end": steps,
                        "drawer_end": drawer_open(core),
                        "eef_end": eef_position(obs),
                        "gripper_qpos_end": gripper_qpos(obs),
                    }
                )
                chunks.append(row)
                actions.append(action.astype(np.float32))
            record = {
                "episode": episode,
                "condition": "residual_clean",
                "style": info["style"],
                "layout": info["layout"],
                "target_drawer": info["target_drawer"],
                "success": success,
                "steps": steps,
                "final_drawer": drawer_open(core),
                "terminated": bool(done),
                "truncated": bool(truncated),
                "chunks": chunks,
            }
            record.update(summarize(record))
            payload["results"].append(record)
            action_arrays[f"ep{episode}_residual_clean"] = np.stack(actions)
            atomic_json(args.out, payload)
            print(
                f"ep={episode} condition=residual_clean success={success} "
                f"steps={steps} drawer={record['final_drawer']:.6f}",
                flush=True,
            )
    finally:
        env.close()
    if file_sha256(args.model) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during residual diagnosis")
    np.savez_compressed(args.actions_out, **action_arrays)
    payload["actions_out"] = str(args.actions_out)
    payload["actions_sha256"] = file_sha256(args.actions_out)
    payload["status"] = "complete"
    payload["elapsed_seconds"] = time.monotonic() - started
    atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "episodes": list(episodes),
                "condition": "x_clean + a_jitter + f_jitter",
                "actions_sha256": payload["actions_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
