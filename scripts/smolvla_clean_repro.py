"""Frozen Clean diagnostic with source provenance; no TTA and no variant search."""

import argparse, ctypes, hashlib, json, os, random, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, torch

CK = Path("/hy-tmp/smolvla_ckpt/HuggingFaceVLA_smolvla_libero")


def ah(x):
    a = np.ascontiguousarray(x)
    return hashlib.sha256(
        str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()
    ).hexdigest()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def axis(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - q[3] * q[3])
    return np.zeros(3) if np.isclose(den, 0) else q[:3] * 2 * np.arccos(q[3]) / den


def main(a):
    import libero, lerobot
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from safetensors import safe_open

    assert "phase3_patch" not in os.environ.get("PYTHONPATH", "")
    assert os.environ["MUJOCO_GL"] == os.environ["PYOPENGL_PLATFORM"] == "egl"
    assert (
        sha(CK / "model.safetensors")
        == "71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f"
    )
    tasknames = json.loads(
        Path("/hy-tmp/patch-diag/runs/smolvla_reproduction_v1/tasks.json").read_text()
    )
    assert len(tasknames) == 10
    if a.pipeline == "modern":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.processor.env_processor import LiberoProcessorStep
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        from lerobot.envs.utils import preprocess_observation
    else:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    model = SmolVLAPolicy.from_pretrained(CK).eval().cuda().requires_grad_(False)
    assert model.config.n_action_steps == 1 and model.config.num_steps == 10
    # Check values, not only matching key names. Normalizer modules differ across versions.
    sd = model.state_dict()
    checks = []
    notin = []
    with safe_open(CK / "model.safetensors", framework="pt", device="cpu") as f:
        for k in f.keys():
            if k not in sd:
                notin.append(k)
                continue
            v = f.get_tensor(k)
            checks.append(torch.equal(sd[k].detach().cpu(), v.to(dtype=sd[k].dtype)))
    assert all(checks) and len(checks) >= 780, (len(checks), notin)
    stats = {}
    for name in [
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
    ]:
        with safe_open(CK / name, framework="pt", device="cpu") as f:
            stats.update({k: f.get_tensor(k) for k in f.keys()})
    if a.pipeline == "modern":
        pre, post = make_pre_post_processors(
            policy_cfg=model.config,
            pretrained_path=CK,
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        envp = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
    else:
        for module, key in [
            (model.normalize_inputs, "observation.state"),
            (model.normalize_targets, "action"),
            (model.unnormalize_outputs, "action"),
        ]:
            buf = getattr(module, "buffer_" + key.replace(".", "_"))
            for kind in ("mean", "std"):
                buf[kind].data.copy_(stats[key + "." + kind].to(buf[kind]))
    a.out.mkdir(parents=True, exist_ok=True)
    task = tasknames[a.task]
    seed = a.task * 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ctypes.CDLL(None).srand(ctypes.c_uint(seed))
    bd = Path(get_libero_path("bddl_files")) / "libero_spatial" / (task + ".bddl")
    ip = (
        Path(get_libero_path("init_states"))
        / "libero_spatial"
        / (task + ".pruned_init")
    )
    init_states = torch.load(ip, map_location="cpu", weights_only=False)
    if not 0 <= a.init_index < len(init_states):
        raise ValueError(f"init index {a.init_index} outside asset with {len(init_states)} states")
    initial = init_states[a.init_index]
    stem = f"{a.pipeline}_{a.assets}_t{a.task}"
    if a.init_index != 0:
        stem += f"_i{a.init_index}"
    # Refuse existing evidence before any rollout or image write.
    assert not (a.out / (stem + ".json")).exists()
    assert not (a.out / (stem + "_initial.npz")).exists()
    if a.assets == "vanilla":
        suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        assert suite.get_num_tasks() == 10
        assert suite.get_task(a.task).name == task
        lang = suite.get_task(a.task).language
    else:
        lang = benchmark.grab_language_from_filename("libero_spatial", task + ".bddl")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bd),
        camera_heights=256,
        camera_widths=256,
        render_gpu_device_id=0,
    )
    start = time.monotonic()
    trace = []
    first = None
    try:
        env.noise = 0
        env.seed(seed)
        env.reset()
        if a.scene_patch != "none":
            assert a.assets == "plus" and a.task == 0
            xml = env.sim.model.get_xml()
            if a.scene_patch in ("textures", "both"):
                old = "/hy-tmp/LIBERO-plus/libero/libero/assets/textures/"
                assert old in xml
                xml = xml.replace(
                    old, "/hy-tmp/LIBERO-vanilla/libero/libero/assets/textures/"
                )
            if a.scene_patch in ("camera", "both"):
                root = ET.fromstring(xml)
                found = False
                for cam in root.iter("camera"):
                    if cam.get("name") == "agentview":
                        cam.set("pos", "0.658613 0 1.61035")
                        cam.set("quat", "0.638018 0.30485 0.30485 0.638018")
                        found = True
                assert found
                xml = ET.tostring(root, encoding="unicode")
            env.reset_from_xml_string(xml)
        effective_xml_sha = hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest()
        obs = env.set_init_state(initial)
        for _ in range(10):
            obs = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))[0]
        for robot in env.robots:
            robot.controller.use_delta = True
        model.reset()
        steps = 0
        success = False
        actions = hashlib.sha256()
        sourcehash = ah(env.sim.get_state().flatten())
        np.savez_compressed(
            a.out / (stem + "_initial.npz"),
            agent=obs["agentview_image"],
            wrist=obs["robot0_eye_in_hand_image"],
            state=np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    axis(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ),
        )
        while steps < a.max_steps:
            rawstate = np.concatenate(
                [
                    obs["robot0_eef_pos"],
                    axis(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                ]
            ).astype(np.float32)
            if a.pipeline == "modern":
                raw = {
                    "pixels": {
                        "image": np.asarray(obs["agentview_image"]),
                        "image2": np.asarray(obs["robot0_eye_in_hand_image"]),
                    },
                    "robot_state": {
                        "eef": {
                            "pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32)[
                                None
                            ],
                            "quat": np.asarray(
                                obs["robot0_eef_quat"], dtype=np.float32
                            )[None],
                        },
                        "gripper": {
                            "qpos": np.asarray(
                                obs["robot0_gripper_qpos"], dtype=np.float32
                            )[None]
                        },
                    },
                }
                v = preprocess_observation(raw)
                v["task"] = [lang]
                v = envp(v)
                beforestate = v["observation.state"].detach().cpu().numpy().reshape(-1)
                np.testing.assert_allclose(beforestate, rawstate, atol=3e-6, rtol=1e-6)
                v = pre(v)
            else:
                image = lambda k: (
                    torch.from_numpy(
                        np.ascontiguousarray(np.asarray(obs[k])[::-1, ::-1])
                    )
                    .permute(2, 0, 1)[None]
                    .float()
                    .cuda()
                    / 255.0
                )
                v = {
                    "observation.images.image": image("agentview_image"),
                    "observation.images.image2": image("robot0_eye_in_hand_image"),
                    "observation.state": torch.from_numpy(rawstate)[None].cuda(),
                    "task": lang,
                }
            with torch.inference_mode():
                act = model.select_action(v)
            if a.pipeline == "modern":
                act = post(act)
            act = act.detach().cpu().numpy().reshape(-1)
            assert act.shape == (7,) and np.isfinite(act).all()
            command = np.clip(act, -1, 1).astype(np.float32)
            actions.update(command.tobytes())
            if steps == 0:
                first = dict(
                    raw_agent=ah(obs["agentview_image"]),
                    raw_wrist=ah(obs["robot0_eye_in_hand_image"]),
                    raw_state=rawstate.tolist(),
                    input_keys=sorted(v),
                    action_unclipped=act.tolist(),
                    action=command.tolist(),
                    language=lang,
                    vision_dtype=str(
                        next(
                            model.model.vlm_with_expert.vlm.model.vision_model.parameters()
                        ).dtype
                    ),
                )
            if steps % 10 == 0:
                trace.append(
                    dict(
                        step=steps,
                        eef=np.asarray(obs["robot0_eef_pos"]).tolist(),
                        action=command.tolist(),
                    )
                )
            obs, reward, done, info = env.step(command)
            steps += 1
            success = bool(env.check_success())
            if success or done:
                break
    finally:
        env.close()
    row = dict(
        scene_patch=a.scene_patch,
        effective_xml_sha256=effective_xml_sha,
        pipeline=a.pipeline,
        asset_tree=a.assets,
        task_id=a.task,
        task=task,
        init_index=a.init_index,
        seed=seed,
        success=success,
        steps=steps,
        max_steps=a.max_steps,
        seconds=time.monotonic() - start,
        libero_module=str(Path(benchmark.__file__).resolve()),
        lerobot_module=(
            str(Path(lerobot.__file__).resolve())
            if lerobot.__file__
            else list(lerobot.__path__)
        ),
        bddl_sha256=sha(bd),
        init_asset_sha256=sha(ip),
        raw_init_sha256=ah(np.asarray(initial)),
        settled_state=sourcehash,
        weights_keys_matched=len(checks),
        checkpoint_keys_unused=notin,
        first=first,
        trace=trace,
        actions_sha256=actions.hexdigest(),
    )
    f = a.out / (stem + ".json")
    assert not f.exists()
    f.write_text(json.dumps(row, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: row[k]
                for k in [
                    "pipeline",
                    "asset_tree",
                    "task_id",
                    "success",
                    "steps",
                    "seconds",
                    "weights_keys_matched",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", choices=("modern", "legacy"), required=True)
    p.add_argument("--assets", choices=("vanilla", "plus"), required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--init-index", type=int, default=0)
    p.add_argument(
        "--scene-patch", choices=("none", "identity", "textures", "camera", "both"), default="none"
    )
    p.add_argument("--max-steps", type=int, default=280)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
