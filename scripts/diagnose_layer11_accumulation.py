"""Diagnose low-action-error closed-loop drift after layer11 interventions.

This is a privileged diagnostic. It compares clean, attention+complete-FFN,
and full clean layer11 block-output replacement on episodes that failed despite
near-zero action errors. It records per-action-group errors, gripper qpos,
EEF/drawer trajectories, and the raw action chunks.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "robocasa"))

from dacsc.corp_data import LinearNormalizer, read_episode_meta  # noqa: E402
from dacsc.corp_robocasa import load_adapted_checkpoint  # noqa: E402
from dacsc.eval_protocol import file_sha256  # noqa: E402
from dacsc.upstream import verify_upstream  # noqa: E402
from scripts.eval_corp_robocasa_sr import policy_inputs  # noqa: E402

CONDITIONS = ("clean", "attention_ffn", "block_output")
ACTION_STEPS = 8
MAX_STEPS = 300
GROUPS = {
    "position": slice(0, 3),
    "rotation": slice(3, 6),
    "gripper": slice(6, 7),
    "base": slice(7, 11),
    "control": slice(11, 12),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--actions-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", default="10,50")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def load_episode(dataset: Path, episode: int) -> tuple[dict, np.ndarray, str]:
    extra = dataset / "extras" / f"episode_{episode:06d}"
    meta = json.loads((extra / "ep_meta.json").read_text())
    with np.load(extra / "states.npz", allow_pickle=False) as state_file:
        state = np.asarray(state_file["states"][0])
    with gzip.open(extra / "model.xml.gz", "rb") as model_file:
        model_xml = model_file.read().decode()
    return meta, state, model_xml


def drawer_open(core) -> float:
    return float(core.drawer.get_door_state(env=core)["door"])


def eef_position(obs) -> list[float]:
    return np.asarray(obs["state.end_effector_position_relative"]).round(5).tolist()


def gripper_qpos(obs) -> list[float]:
    return np.asarray(obs["state.gripper_qpos"]).round(6).tolist()


def sample_action(model, images, state, lang, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        action = model.sample_action(images, state, lang)[0].float().cpu().numpy()
    if (
        action.ndim != 2
        or action.shape[1] != 12
        or len(action) < ACTION_STEPS
        or not np.isfinite(action).all()
    ):
        raise ValueError("policy returned an invalid RoboCasa action chunk")
    return action


def capture_clean_branches(block, visual, images) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
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
        block.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__(
                "block_output", output.detach().clone()
            )
        ),
    ]
    try:
        with torch.inference_mode():
            visual(images)
    finally:
        for hook in hooks:
            hook.remove()
    if set(captured) != {"attention", "ffn", "block_output"}:
        raise RuntimeError("clean branch capture is incomplete")
    return captured


def install_hooks(block, condition: str, captured: dict[str, torch.Tensor]):
    if condition == "clean":
        return []
    hooks = []
    if condition in ("attention_ffn", "block_output"):
        hooks.append(
            block.ls1.register_forward_hook(
                lambda _module, _inputs, output: captured["attention"]
                .to(device=output.device, dtype=output.dtype)
                .clone()
            )
        )
    if condition == "attention_ffn":
        hooks.append(
            block.ls2.register_forward_hook(
                lambda _module, _inputs, output: captured["ffn"]
                .to(device=output.device, dtype=output.dtype)
                .clone()
            )
        )
    elif condition == "block_output":
        hooks.append(
            block.register_forward_hook(
                lambda _module, _inputs, output: captured["block_output"]
                .to(device=output.device, dtype=output.dtype)
                .clone()
            )
        )
    else:
        raise ValueError(f"unknown condition: {condition}")
    return hooks


def action_metrics(action: np.ndarray, clean: np.ndarray) -> dict:
    return {
        "mse_clean": float(np.mean((action - clean) ** 2)),
        "cos_clean": float(
            F.cosine_similarity(
                torch.from_numpy(action).flatten(),
                torch.from_numpy(clean).flatten(),
                dim=0,
            )
        ),
        "groups_mse_clean": {
            name: float(np.mean((action[..., indices] - clean[..., indices]) ** 2))
            for name, indices in GROUPS.items()
        },
        "max_abs_clean": float(np.max(np.abs(action - clean))),
        "action_max_abs": float(np.max(np.abs(action))),
    }


def run_condition(
    *,
    episode: int,
    condition: str,
    env,
    core,
    model,
    norm,
    device,
    meta: dict,
    initial_state: np.ndarray,
    model_xml: str,
    reset_to,
    convert_action,
    visual,
    block,
) -> tuple[dict, list[np.ndarray]]:
    reset_to(
        core,
        {
            "model": model_xml,
            "states": initial_state,
            "ep_meta": json.dumps(meta),
        },
    )
    obs = env.get_observation(core._get_observations(force_update=True))
    success = False
    done = False
    truncated = False
    steps = 0
    chunks = []
    raw_actions = []
    while not (success or done or truncated) and steps < MAX_STEPS:
        clean_images, state, lang = policy_inputs(
            obs, meta["lang"], norm, device, 0.0, "none"
        )
        jitter_images, jitter_state, jitter_lang = policy_inputs(
            obs, meta["lang"], norm, device, 0.0, "color_jitter"
        )
        if not torch.equal(state, jitter_state) or lang != jitter_lang:
            raise RuntimeError("clean and jitter policy inputs are not paired")

        policy_seed = 100000 + episode * 1000 + len(chunks)
        clean_action = sample_action(model, clean_images, state, lang, policy_seed)
        captured = capture_clean_branches(
            block,
            visual,
            torch.cat([clean_images[view] for view in model.view_names]),
        )
        hooks = install_hooks(block, condition, captured)
        try:
            if condition == "clean":
                action = clean_action
            else:
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
            obs, _, done, truncated, info = env.step(
                convert_action(np.clip(raw_action, -1, 1).astype(np.float32))
            )
            steps += 1
            success = bool(info["success"])
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
        raw_actions.append(action.astype(np.float32))
    return (
        {
            "episode": episode,
            "condition": condition,
            "success": success,
            "steps": steps,
            "final_drawer": drawer_open(core),
            "terminated": bool(done),
            "truncated": bool(truncated),
            "chunks": chunks,
        },
        raw_actions,
    )


def summarize(record: dict) -> dict:
    chunks = record["chunks"]
    if not chunks:
        raise ValueError("rollout has no action chunks")
    return {
        "num_chunks": len(chunks),
        "mse_clean_mean": float(np.mean([chunk["mse_clean"] for chunk in chunks])),
        "mse_clean_max": float(np.max([chunk["mse_clean"] for chunk in chunks])),
        "cos_clean_mean": float(np.mean([chunk["cos_clean"] for chunk in chunks])),
        "cos_clean_min": float(np.min([chunk["cos_clean"] for chunk in chunks])),
        "groups_mse_clean_mean": {
            group: float(
                np.mean([chunk["groups_mse_clean"][group] for chunk in chunks])
            )
            for group in GROUPS
        },
        "groups_mse_clean_max": {
            group: float(
                np.max([chunk["groups_mse_clean"][group] for chunk in chunks])
            )
            for group in GROUPS
        },
    }


def trajectory_comparison(clean: dict, evaluated: dict) -> dict:
    clean_chunks = clean["chunks"]
    chunks = evaluated["chunks"]
    common = min(len(clean_chunks), len(chunks))
    distances = [
        float(
            np.linalg.norm(
                np.asarray(chunks[index]["eef_end"], dtype=np.float64)
                - np.asarray(clean_chunks[index]["eef_end"], dtype=np.float64)
            )
        )
        for index in range(common)
    ]
    drawer_differences = [
        abs(chunks[index]["drawer_end"] - clean_chunks[index]["drawer_end"])
        for index in range(common)
    ]
    return {
        "common_chunks": common,
        "eef_divergence_ge_1cm_chunk": next(
            (index for index, value in enumerate(distances) if value >= 0.01),
            None,
        ),
        "eef_divergence_ge_5cm_chunk": next(
            (index for index, value in enumerate(distances) if value >= 0.05),
            None,
        ),
        "drawer_divergence_ge_0_01_chunk": next(
            (index for index, value in enumerate(drawer_differences) if value >= 0.01),
            None,
        ),
        "final_eef_distance": distances[-1],
        "max_eef_distance": max(distances),
    }


def main() -> int:
    args = parse_args()
    if args.out.exists() or args.actions_out.exists():
        raise FileExistsError("refusing to overwrite diagnostic outputs")
    upstream = verify_upstream()
    from robocasa.models.scenes.scene_registry import LayoutType
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.utils.env_utils import convert_action
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    episodes = tuple(int(item) for item in args.episodes.split(","))
    if not episodes or len(episodes) != len(set(episodes)):
        raise ValueError("episodes must be a non-empty unique list")
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
        "schema": "dacsc_layer11_accumulation_diagnostic_v1",
        "started_at_epoch": started,
        "scope": {
            "episodes": list(episodes),
            "conditions": list(CONDITIONS),
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
            episode_records = {}
            for condition in CONDITIONS:
                record, actions = run_condition(
                    episode=episode,
                    condition=condition,
                    env=env,
                    core=core,
                    model=model,
                    norm=norm,
                    device=device,
                    meta=info["meta"],
                    initial_state=info["initial_state"],
                    model_xml=info["model_xml"],
                    reset_to=reset_to,
                    convert_action=convert_action,
                    visual=visual,
                    block=block,
                )
                record.update(
                    {
                        "style": info["style"],
                        "layout": info["layout"],
                        "target_drawer": info["target_drawer"],
                        **summarize(record),
                    }
                )
                episode_records[condition] = record
                action_arrays[f"ep{episode}_{condition}"] = np.stack(actions)
                payload["results"].append(record)
                atomic_json(args.out, payload)
                print(
                    f"ep={episode} condition={condition} success={record['success']} "
                    f"steps={record['steps']} drawer={record['final_drawer']:.6f}",
                    flush=True,
                )
            for condition in CONDITIONS[1:]:
                episode_records[condition]["trajectory_vs_clean"] = (
                    trajectory_comparison(
                        episode_records["clean"], episode_records[condition]
                    )
                )
            atomic_json(args.out, payload)
    finally:
        env.close()
    if file_sha256(args.model) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during diagnosis")
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
                "conditions": list(CONDITIONS),
                "actions_sha256": payload["actions_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
