"""Frozen scene/state manifests and RNG control for paired RoboCasa rollouts."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_episode(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported manifest value: {type(value).__name__}")


def prepare_episode(
    env,
    path: Path,
    *,
    task: str,
    styles: list[int],
    layouts: list[int],
    episode: int,
    seed: int,
    mode: str,
    reset_to,
):
    """Always restore the recorded model before its state, including on creation.

    reset_to is RoboCasa's official playback helper, injected to allow local tests
    without MuJoCo/assets. Creation is explicit and refuses existing manifests.
    """
    core = env.env
    contract = {
        "task": task,
        "allowed_styles": styles,
        "allowed_layouts": layouts,
        "episode": episode,
        "environment_seed": seed,
        "policy_seed": seed,
    }
    if mode == "create":
        if path.exists():
            raise FileExistsError(path)
        seed_episode(seed)
        env.reset(seed=seed)
        meta = core.get_ep_meta()
        actual_style = int(meta["style_id"])
        if actual_style not in styles:
            raise ValueError("sampled style is outside the requested split")
        if int(meta["layout_id"]) not in layouts:
            raise ValueError("sampled layout is outside the pretrain layout group")
        manifest = {
            "format": "robocasa_eval_episode_v1",
            **contract,
            "actual_style": actual_style,
            "actual_layout": int(meta["layout_id"]),
            "ep_meta": meta,
            "model_xml": core.sim.model.get_xml(),
            "sim_state": core.sim.get_state().flatten().tolist(),
            "environment_rng": core.rng.bit_generator.state,
        }
        encoded = json.dumps(manifest, sort_keys=True, default=_json_value).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb") as stream:
            stream.write(encoded)
        manifest = json.loads(encoded)
    elif mode == "replay":
        with gzip.open(path, "rb") as stream:
            encoded = stream.read()
        manifest = json.loads(encoded)
    else:
        raise ValueError(f"unknown manifest mode: {mode}")
    if manifest["format"] != "robocasa_eval_episode_v1":
        raise ValueError("unsupported episode manifest")
    if any(manifest[key] != value for key, value in contract.items()):
        raise ValueError(f"episode manifest does not match requested protocol: {path}")
    seed_episode(seed)
    state = np.asarray(manifest["sim_state"], dtype=np.float64)
    reset_to(
        core,
        {
            "model": manifest["model_xml"],
            "states": state,
            "ep_meta": json.dumps(manifest["ep_meta"]),
        },
    )
    restored = core.sim.get_state().flatten()
    if restored.shape != state.shape or not np.allclose(
        restored, state, atol=1e-9, rtol=0
    ):
        raise RuntimeError("simulator state does not match frozen initial state")
    actual = core.get_ep_meta()
    if (
        int(actual["style_id"]) != manifest["actual_style"]
        or int(actual["layout_id"]) != manifest["actual_layout"]
    ):
        raise RuntimeError("restored style/layout does not match manifest")
    core.rng.bit_generator.state = manifest["environment_rng"]
    obs = env.get_observation(core._get_observations(force_update=True))
    # Reset policy noise after restoration, independent of reset's RNG consumption.
    seed_episode(manifest["policy_seed"])
    return obs, manifest, hashlib.sha256(encoded).hexdigest()


def rollout_episode(
    env,
    model,
    norm,
    device,
    obs,
    lang,
    *,
    policy_inputs,
    convert_action,
    max_steps: int,
    action_steps: int,
    blur_sigma: float,
    shift: str,
) -> dict:
    success, steps, done, truncated = False, 0, False, False
    while not (success or done or truncated) and steps < max_steps:
        images, state, instruction = policy_inputs(
            obs, lang, norm, device, blur_sigma, shift
        )
        with torch.no_grad():
            chunk = (
                model.sample_action(images, state, instruction)[0].float().cpu().numpy()
            )
        if (
            chunk.ndim != 2
            or chunk.shape[1] != 12
            or len(chunk) < action_steps
            or not np.isfinite(chunk).all()
        ):
            raise ValueError(
                "policy returned an invalid/empty/nonfinite RoboCasa action chunk"
            )
        for action in chunk[:action_steps]:
            obs, _, done, truncated, info = env.step(
                convert_action(np.clip(action, -1, 1).astype(np.float32))
            )
            steps += 1
            success = bool(info["success"])
            if success or done or truncated or steps >= max_steps:
                break
    return {
        "success": success,
        "steps": steps,
        "terminated": bool(done),
        "truncated": bool(truncated),
    }
