"""Validate layer11 attention-path stability on fixed OpenDrawer episodes.

The candidate pool and eligibility rule are fixed before any branch
intervention. Clean/jitter qualification runs first; only the first four
eligible episodes enter the privileged attention-only and attention+FFN
closed-loop checks.
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

CANDIDATE_POOL = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
REQUIRED_EPISODES = 4
QUALIFICATION_MAX_STEPS = 750
INTERVENTION_MAX_STEPS = 300
ACTION_STEPS = 8
SOURCE_HOLDOUT_RESIDUE = 9
TRAIN_STYLE_RANGE = (11, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def load_state(dataset: Path, episode: int) -> tuple[dict, np.ndarray, str]:
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
        "gripper_mse": float(np.mean((action[:, 6] - clean[:, 6]) ** 2)),
    }


def branch_capture(block, visual, clean_images) -> dict[str, torch.Tensor]:
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
    ]
    try:
        with torch.inference_mode():
            visual(clean_images)
    finally:
        for hook in hooks:
            hook.remove()
    if set(captured) != {"attention", "ffn"}:
        raise RuntimeError("layer11 branch capture did not produce attention and FFN")
    return captured


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


def install_branch_hooks(block, condition: str, captured: dict[str, torch.Tensor]):
    if condition == "attention":
        names = ("attention",)
    elif condition == "attention_ffn":
        names = ("attention", "ffn")
    else:
        raise ValueError(f"unknown intervention condition: {condition}")
    hooks = []
    for name in names:
        module = block.ls1 if name == "attention" else block.ls2

        def replace(_module, _inputs, output, *, branch=name):
            return captured[branch].to(device=output.device, dtype=output.dtype).clone()

        hooks.append(module.register_forward_hook(replace))
    return hooks


def run_rollout(
    *,
    episode: int,
    condition: str,
    env,
    core,
    model,
    norm,
    device,
    dataset: Path,
    meta: dict,
    initial_state: np.ndarray,
    model_xml: str,
    reset_to,
    convert_action,
    max_steps: int,
    visual,
    block,
) -> dict:
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
    while not (success or done or truncated) and steps < max_steps:
        clean_images, state, lang = policy_inputs(
            obs, meta["lang"], norm, device, 0.0, "none"
        )
        if condition == "clean":
            evaluated_images = clean_images
        else:
            evaluated_images, evaluated_state, evaluated_lang = policy_inputs(
                obs, meta["lang"], norm, device, 0.0, "color_jitter"
            )
            if not torch.equal(state, evaluated_state) or lang != evaluated_lang:
                raise RuntimeError("clean and jitter policy inputs are not paired")
            evaluated_state, evaluated_lang = state, lang

        policy_seed = 100000 + episode * 1000 + len(chunks)
        clean_action = sample_action(model, clean_images, state, lang, policy_seed)
        captured = None
        if condition in ("attention", "attention_ffn"):
            flat_clean = torch.cat([clean_images[view] for view in model.view_names])
            captured = branch_capture(block, visual, flat_clean)
            hooks = install_branch_hooks(block, condition, captured)
            try:
                action = sample_action(
                    model,
                    evaluated_images,
                    evaluated_state,
                    evaluated_lang,
                    policy_seed,
                )
            finally:
                for hook in hooks:
                    hook.remove()
        elif condition == "color_jitter":
            action = sample_action(
                model,
                evaluated_images,
                evaluated_state,
                evaluated_lang,
                policy_seed,
            )
        else:
            action = clean_action

        row = {
            "chunk": len(chunks),
            "step_start": steps,
            **action_metrics(action, clean_action),
            "drawer_start": drawer_open(core),
            "eef_start": eef_position(obs),
        }
        for raw_action in action[:ACTION_STEPS]:
            obs, _, done, truncated, info = env.step(
                convert_action(np.clip(raw_action, -1, 1).astype(np.float32))
            )
            steps += 1
            success = bool(info["success"])
            if success or done or truncated or steps >= max_steps:
                break
        row.update(
            {
                "step_end": steps,
                "drawer_end": drawer_open(core),
                "eef_end": eef_position(obs),
            }
        )
        chunks.append(row)
    return {
        "episode": episode,
        "condition": condition,
        "success": success,
        "steps": steps,
        "final_drawer": drawer_open(core),
        "terminated": bool(done),
        "truncated": bool(truncated),
        "chunks": chunks,
    }


def rollout_stats(record: dict) -> dict:
    chunks = record["chunks"]
    if not chunks:
        raise ValueError("completed rollout must contain at least one action chunk")
    mse = [chunk["mse_clean"] for chunk in chunks]
    cosine = [chunk["cos_clean"] for chunk in chunks]
    gripper = [chunk["gripper_mse"] for chunk in chunks]
    return {
        "chunks": len(chunks),
        "chunk_mse_clean_mean": float(np.mean(mse)),
        "chunk_mse_clean_max": float(np.max(mse)),
        "chunk_cos_clean_mean": float(np.mean(cosine)),
        "chunk_cos_clean_min": float(np.min(cosine)),
        "chunk_gripper_mse_mean": float(np.mean(gripper)),
        "chunk_gripper_mse_max": float(np.max(gripper)),
    }


def main() -> int:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {args.out}")
    upstream = verify_upstream()
    from robocasa.models.scenes.scene_registry import LayoutType
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.utils.env_utils import convert_action
    from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv

    metadata = read_episode_meta(args.dataset)
    candidate_meta = {}
    for episode in CANDIDATE_POOL:
        if episode not in metadata:
            raise ValueError(f"candidate episode is absent: {episode}")
        item = metadata[episode]
        if not TRAIN_STYLE_RANGE[0] <= item["style_id"] <= TRAIN_STYLE_RANGE[1]:
            raise ValueError(f"candidate episode {episode} is outside train styles")
        if episode % 10 == SOURCE_HOLDOUT_RESIDUE:
            raise ValueError(f"candidate episode {episode} is source holdout")
        episode_meta, state, model_xml = load_state(args.dataset, episode)
        fixture_refs = episode_meta.get("fixture_refs")
        if not isinstance(fixture_refs, dict) or not fixture_refs.get("drawer"):
            raise ValueError(f"episode {episode} lacks a drawer fixture ref")
        candidate_meta[episode] = {
            "style": item["style_id"],
            "layout": item["layout_id"],
            "target_drawer": fixture_refs["drawer"],
            "ep_meta": episode_meta,
            "state": state,
            "model_xml": model_xml,
        }

    checkpoint_sha256 = file_sha256(args.model)
    device = torch.device(args.device)
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    if model.action_horizon < ACTION_STEPS:
        raise ValueError("checkpoint action horizon is shorter than action_steps")
    norm = LinearNormalizer(mean, std)
    visual = model.representation_extractor.visual_encoder
    block = visual.backbone.blocks[11]
    styles = sorted({item["style"] for item in candidate_meta.values()})
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
        "schema": "dacsc_layer11_branch_stability_v1",
        "started_at_epoch": started,
        "candidate_policy": {
            "rule": "fixed ascending training-split OpenDrawer episodes ending in zero",
            "ordered_pool": list(CANDIDATE_POOL),
            "required_episodes": REQUIRED_EPISODES,
            "eligibility": {
                "clean": "success at <=300 steps",
                "color_jitter": "failure at 750 steps",
            },
            "selection_uses_branch_results": False,
            "scope": "mechanism-validation holdout, not official source holdout",
        },
        "protocol": {
            "task": "OpenDrawer",
            "scene_split": "pretrain",
            "initial_state": "official training episode extras/episode_*/states.npz frame0",
            "action_steps": ACTION_STEPS,
            "qualification_max_steps": QUALIFICATION_MAX_STEPS,
            "intervention_max_steps": INTERVENTION_MAX_STEPS,
            "policy_seed_formula": "100000 + episode * 1000 + chunk",
            "checkpoint": str(args.model),
            "checkpoint_sha256": checkpoint_sha256,
            "upstream": upstream,
            "clean_reference_is_privileged": True,
            "deployable_method": False,
        },
        "qualification": [],
        "selected_episodes": [],
        "interventions": [],
        "status": "running_qualification",
    }
    atomic_json(args.out, payload)
    selected = []
    try:
        for episode in CANDIDATE_POOL:
            info = candidate_meta[episode]
            record = {
                "episode": episode,
                "style": info["style"],
                "layout": info["layout"],
                "target_drawer": info["target_drawer"],
            }
            clean = run_rollout(
                episode=episode,
                condition="clean",
                env=env,
                core=core,
                model=model,
                norm=norm,
                device=device,
                dataset=args.dataset,
                meta=info["ep_meta"],
                initial_state=info["state"],
                model_xml=info["model_xml"],
                reset_to=reset_to,
                convert_action=convert_action,
                max_steps=QUALIFICATION_MAX_STEPS,
                visual=visual,
                block=block,
            )
            record["clean"] = clean
            record["clean_eligible"] = clean["success"] and clean["steps"] <= 300
            if record["clean_eligible"]:
                jitter = run_rollout(
                    episode=episode,
                    condition="color_jitter",
                    env=env,
                    core=core,
                    model=model,
                    norm=norm,
                    device=device,
                    dataset=args.dataset,
                    meta=info["ep_meta"],
                    initial_state=info["state"],
                    model_xml=info["model_xml"],
                    reset_to=reset_to,
                    convert_action=convert_action,
                    max_steps=QUALIFICATION_MAX_STEPS,
                    visual=visual,
                    block=block,
                )
                record["color_jitter"] = jitter
                record["eligible"] = not jitter["success"] and (
                    jitter["steps"] == QUALIFICATION_MAX_STEPS
                )
            else:
                record["color_jitter"] = None
                record["eligible"] = False
            payload["qualification"].append(record)
            if record["eligible"]:
                selected.append(episode)
                payload["selected_episodes"] = selected
            atomic_json(args.out, payload)
            print(
                f"qualification ep={episode} style={info['style']} "
                f"layout={info['layout']} clean={clean['success']}@{clean['steps']} "
                f"jitter="
                f"{None if record['color_jitter'] is None else record['color_jitter']['success']}"
                f" eligible={record['eligible']}",
                flush=True,
            )
            if len(selected) == REQUIRED_EPISODES:
                break
        if len(selected) != REQUIRED_EPISODES:
            payload["status"] = "insufficient_eligible_episodes"
            payload["elapsed_seconds"] = time.monotonic() - started
            atomic_json(args.out, payload)
            print(json.dumps(payload["selected_episodes"]), flush=True)
            return 2

        payload["status"] = "running_interventions"
        atomic_json(args.out, payload)
        for episode in selected:
            info = candidate_meta[episode]
            for condition in ("attention", "attention_ffn"):
                result = run_rollout(
                    episode=episode,
                    condition=condition,
                    env=env,
                    core=core,
                    model=model,
                    norm=norm,
                    device=device,
                    dataset=args.dataset,
                    meta=info["ep_meta"],
                    initial_state=info["state"],
                    model_xml=info["model_xml"],
                    reset_to=reset_to,
                    convert_action=convert_action,
                    max_steps=INTERVENTION_MAX_STEPS,
                    visual=visual,
                    block=block,
                )
                payload["interventions"].append(
                    {
                        "episode": episode,
                        "style": info["style"],
                        "layout": info["layout"],
                        "target_drawer": info["target_drawer"],
                        "condition": condition,
                        "success": result["success"],
                        "steps": result["steps"],
                        "final_drawer": result["final_drawer"],
                        **rollout_stats(result),
                        "chunks": result["chunks"],
                    }
                )
                atomic_json(args.out, payload)
                print(
                    f"intervention ep={episode} condition={condition} "
                    f"success={result['success']} steps={result['steps']} "
                    f"drawer={result['final_drawer']:.6f}",
                    flush=True,
                )
    finally:
        env.close()
    if file_sha256(args.model) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during validation")
    payload["status"] = "complete"
    payload["elapsed_seconds"] = time.monotonic() - started
    payload["attention_successes"] = sum(
        item["success"] for item in payload["interventions"] if item["condition"] == "attention"
    )
    payload["attention_ffn_successes"] = sum(
        item["success"]
        for item in payload["interventions"]
        if item["condition"] == "attention_ffn"
    )
    payload["episodes"] = len(selected)
    atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "selected_episodes": selected,
                "attention_successes": payload["attention_successes"],
                "attention_ffn_successes": payload["attention_ffn_successes"],
                "episodes": payload["episodes"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
