"""Verify the live CoRP DINO patch interface and identity equivalence.

This is a source-Clean, single-state diagnostic. It performs no training,
adaptation, target evaluation, or rollout beyond environment settling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos/CoRP"))
sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))

from corp import load_policy  # noqa: E402
from corp.preprocessing import prepare_observation  # noqa: E402
from corp.representation.extractor import Representation  # noqa: E402
from evaluate_libero import DUMMY_ACTION, SETTLING_STEPS, policy_observation  # noqa: E402
import libero.libero.envs.bddl_utils as BDDLUtils  # noqa: E402
from target_only_action_validation import configure  # noqa: E402
from dacsc.differentiable_policy import sample_from_z0  # noqa: E402
from dacsc.patch_diagnostics import validate_patch_dict  # noqa: E402
from dacsc.tta_protocol import load_split, role_ids, validate_split_identity  # noqa: E402


def _shape(value):
    return [int(x) for x in value.shape]


def _metadata_candidates(encoder):
    items = []
    for name, module in encoder.named_modules():
        record = {"module": name or "<root>", "class": type(module).__name__}
        for attr in ("patch_size", "num_prefix_tokens", "num_register_tokens", "num_registers"):
            value = getattr(module, attr, None)
            if isinstance(value, (int, float, str, tuple, list)):
                record[attr] = value
        for attr in ("cls_token", "register_tokens"):
            value = getattr(module, attr, None)
            if value is not None and hasattr(value, "shape"):
                record[attr + "_shape"] = _shape(value)
        if isinstance(module, torch.nn.Conv2d):
            record["kernel_size"] = list(module.kernel_size)
            record["stride"] = list(module.stride)
            record["padding"] = list(module.padding)
            record["dilation"] = list(module.dilation)
        if len(record) > 2:
            items.append(record)
    return items


def _make_rep(visual, language_tokens, pooled_language, state_token):
    return Representation(
        tokens=torch.cat(
            [visual, language_tokens.to(visual.dtype), state_token], dim=1
        ),
        pooled_language=pooled_language,
        action_dtype=visual.dtype,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--lighting-id", type=int, default=2292)
    parser.add_argument("--state-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    parser.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    os.environ["MUJOCO_GL"] = "osmesa"
    configure(args)
    from libero.libero.envs import OffScreenRenderEnv

    matrix = json.loads(args.matrix.read_text())
    task = next(x for x in matrix["base_tasks"] if x["suite"] == args.suite)
    split = load_split(args.split_json)
    validate_split_identity(split, args.suite, task.get("base_task") or "", args.lighting_id)
    fit_ids = role_ids(split, "source_fit")
    state_id = fit_ids[0] if args.state_id is None else int(args.state_id)
    if state_id not in fit_ids:
        raise ValueError(f"Phase 0 state {state_id} is not in source_fit: {fit_ids}")

    variant = next(
        v for v in task["variants"]["lighting"]
        if int(v["classification_id"]) == int(args.lighting_id)
    )
    states = torch.load(task["state_split"]["source_file"], map_location="cpu", weights_only=False)
    language = BDDLUtils.get_problem_info(variant["base_bddl"])["language_instruction"]
    device = torch.device(args.device)
    policy = load_policy(args.checkpoint, device=device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    env = OffScreenRenderEnv(
        bddl_file_name=variant["base_bddl"],
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=args.render_gpu,
    )
    try:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(states[state_id])
        for _ in range(SETTLING_STEPS):
            obs, _, done, _ = env.step(DUMMY_ACTION)
            if done:
                raise RuntimeError(f"episode ended during settling for state {state_id}")

        policy_obs = policy_observation(obs, language, policy)
        prepared = prepare_observation(
            policy_obs,
            config=policy.config,
            normalizer=policy.normalizer,
            device=device,
        )
        extractor = policy.representation_extractor
        views = tuple(extractor.view_names)
        image_shapes = {view: _shape(prepared["images"][view]) for view in views}
        visual_input = torch.cat([prepared["images"][view] for view in views], dim=0)
        trace = {"input_shape": None, "output_shape": None}

        def pre_hook(_module, inputs):
            if inputs and hasattr(inputs[0], "shape"):
                trace["input_shape"] = _shape(inputs[0])

        def post_hook(_module, _inputs, output):
            if hasattr(output, "shape"):
                trace["output_shape"] = _shape(output)

        hooks = []
        if isinstance(extractor.visual_encoder, torch.nn.Module):
            hooks = [
                extractor.visual_encoder.register_forward_pre_hook(pre_hook),
                extractor.visual_encoder.register_forward_hook(post_hook),
            ]
        try:
            with torch.no_grad():
                raw_tokens = extractor.visual_encoder(visual_input)
        finally:
            for hook in hooks:
                hook.remove()

        if raw_tokens.ndim != 3:
            raise RuntimeError(
                f"visual_encoder must return [B,N,C], got {_shape(raw_tokens)}"
            )
        batch_size = int(prepared["images"][views[0]].shape[0])
        patch_dict = {
            view: raw_tokens[i * batch_size:(i + 1) * batch_size]
            for i, view in enumerate(views)
        }
        patch_shapes = validate_patch_dict(patch_dict, views)

        with torch.no_grad():
            language_tokens, pooled_language = extractor.language_encoder(
                prepared["instruction"]
            )
            state_token = extractor.state_encoder(prepared["state"])
            visual_clean_path = extractor.cross_attention_resampler(
                patch_dict, pooled_language
            )
            identity_patch_dict = {k: v.clone() for k, v in patch_dict.items()}
            visual_identity_path = extractor.cross_attention_resampler(
                identity_patch_dict, pooled_language
            )
            rep_clean = _make_rep(
                visual_clean_path, language_tokens, pooled_language, state_token
            )
            rep_identity = _make_rep(
                visual_identity_path, language_tokens.clone(),
                pooled_language.clone(), state_token.clone()
            )
            generator = policy.action_generator
            z0 = torch.randn(
                (batch_size, generator.action_horizon, generator.action_dim),
                device=device,
                dtype=next(policy.parameters()).dtype,
            )
            action_clean = sample_from_z0(policy, rep_clean, z0)
            action_identity = sample_from_z0(policy, rep_identity, z0)

        patch_error = max(
            float((patch_dict[k] - identity_patch_dict[k]).abs().max().item())
            for k in views
        )
        resampler_error = float(
            (visual_clean_path - visual_identity_path).abs().max().item()
        )
        representation_error = float(
            (rep_clean.tokens - rep_identity.tokens).abs().max().item()
        )
        action_error = float((action_clean - action_identity).abs().max().item())

        input_hw = list(visual_input.shape[-2:])
        conv_candidates = [
            item for item in _metadata_candidates(extractor.visual_encoder)
            if "kernel_size" in item and item["kernel_size"][0] == item["kernel_size"][1]
            and item["stride"][0] == item["stride"][1]
        ]
        grid_candidates = []
        for item in conv_candidates:
            kh, kw = item["kernel_size"]
            sh, sw = item["stride"]
            ph, pw = item["padding"]
            dh, dw = item["dilation"]
            gh = (input_hw[0] + 2 * ph - dh * (kh - 1) - 1) // sh + 1
            gw = (input_hw[1] + 2 * pw - dw * (kw - 1) - 1) // sw + 1
            n_out = int(raw_tokens.shape[1])
            grid_candidates.append({
                "module": item["module"],
                "kernel_size": [kh, kw],
                "stride": [sh, sw],
                "expected_grid_hw": [int(gh), int(gw)],
                "expected_grid_tokens": int(gh * gw),
                "actual_tokens_per_view": n_out,
                "actual_minus_grid": int(n_out - gh * gw),
            })

        report = {
            "schema": "corp_dino_patch_interface_v1",
            "state_id": state_id,
            "role": "source_fit",
            "suite": args.suite,
            "lighting_id": args.lighting_id,
            "checkpoint": str(args.checkpoint),
            "view_names": list(views),
            "prepared_image_shapes": image_shapes,
            "visual_encoder_class": type(extractor.visual_encoder).__name__,
            "visual_encoder_hook": trace,
            "patch_shapes": patch_shapes,
            "tokens_per_view": int(raw_tokens.shape[1]),
            "embedding_dim": int(raw_tokens.shape[2]),
            "input_hw": input_hw,
            "patch_metadata_candidates": _metadata_candidates(extractor.visual_encoder),
            "patch_grid_candidates": grid_candidates,
            "identity_errors": {
                "patch_max_abs": patch_error,
                "resampler_max_abs": resampler_error,
                "representation_max_abs": representation_error,
                "action_max_abs": action_error,
            },
            "identity_threshold": 1e-6,
            "identity_pass": bool(max(
                patch_error, resampler_error, representation_error, action_error
            ) < 1e-6),
            "z0_shape": _shape(z0),
            "action_shape": _shape(action_clean),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if not report["identity_pass"]:
            raise RuntimeError("identity transform failed the 1e-6 equivalence check")
    finally:
        env.close()


if __name__ == "__main__":
    main()
