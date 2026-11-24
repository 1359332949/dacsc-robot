"""A diagnostic restricted to 20 named Clean starts. There is no target or adaptation mode."""

import argparse, ctypes, hashlib, json, os, random, sys
from pathlib import Path
import numpy as np, torch
from egl_shallow_block3_closed_loop import ROOT, LIB, CORP, CHECKPOINT
from dacsc.static_visual_v2 import array_hash, file_hash, seed_for

OBJECTS = "alphabet_soup bbq_sauce butter chocolate_pudding cream_cheese ketchup milk orange_juice salad_dressing tomato_sauce".split()


def main(a):
    from evaluate_libero import (
        DUMMY_ACTION,
        SETTLING_STEPS,
        is_success,
        policy_observation,
    )
    from target_only_action_validation import sample_z, visual_from_obs
    from corp import load_policy
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.benchmark import grab_language_from_filename

    assert (
        os.environ["MUJOCO_GL"] == os.environ["PYOPENGL_PLATFORM"] == "egl"
        and 0 <= a.shard < 4
    )
    ck = CORP / "checkpoints/corp_libero.pt"
    assert file_hash(ck) == CHECKPOINT
    model = load_policy(ck, device="cuda:0").eval().requires_grad_(False)
    a.out.mkdir(exist_ok=True, parents=True)
    for ob, ix in [(v, j) for v in OBJECTS for j in (8, 9)][a.shard :: 4]:
        t = f"pick_up_the_{ob}_and_place_it_in_the_basket"
        b = LIB / "libero/libero/bddl_files/libero_object" / (t + ".bddl")
        ip = LIB / "libero/libero/init_files/libero_object" / (t + ".pruned_init")
        state = torch.load(ip, map_location="cpu", weights_only=False)[ix]
        f = a.out / f"{ob}_{ix}.json"
        assert not f.exists() and SETTLING_STEPS == 10
        es = seed_for("block3_object_smoke_v1", t, ix, "env")
        ps = seed_for("block3_object_smoke_v1", t, ix, "policy")
        random.seed(es)
        np.random.seed(es)
        torch.manual_seed(es)
        ctypes.CDLL(None).srand(ctypes.c_uint(es))
        gen = torch.Generator(device="cuda:0").manual_seed(ps)
        lang = grab_language_from_filename("libero_object", t + ".bddl")
        env = OffScreenRenderEnv(
            bddl_file_name=str(b),
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=0,
        )
        steps = 0
        try:
            assert int(env.noise) == 0
            env.noise = 0
            env.seed(es)
            env.reset()
            obs = env.set_init_state(state)
            for _ in range(SETTLING_STEPS):
                obs = env.step(DUMMY_ACTION)[0]
            sim = array_hash(env.sim.get_state().flatten())
            images = [
                array_hash(np.asarray(obs[k]))
                for k in ("agentview_image", "robot0_eye_in_hand_image")
            ]
            action = hashlib.sha256()
            with torch.no_grad():
                while steps < 220:
                    _, rep = visual_from_obs(
                        model, policy_observation(obs, lang, model)
                    )
                    z = torch.randn((1, 12, 7), device="cuda:0", generator=gen)
                    plan = (
                        model.normalizer.unnormalize_action(sample_z(model, rep, z))[0]
                        .clamp(-1, 1)
                        .cpu()
                        .numpy()
                    )
                    assert plan.shape == (12, 7) and np.isfinite(plan).all()
                    for ac in plan:
                        action.update(ac.tobytes())
                        obs, reward, done, info = env.step(ac)
                        steps += 1
                        win = bool(is_success(env, info, reward, done))
                        if done or win or steps == 220:
                            break
                    if done or win:
                        break
        finally:
            env.close()
        row = dict(
            suite="libero_object",
            role="compatibility_only_exclude_from_scoring",
            task=t,
            init_index=ix,
            raw_init_hash=array_hash(np.asarray(state)),
            settled_state_sha256=sim,
            pixels=images,
            language=lang,
            bddl_sha256=file_hash(b),
            init_asset_sha256=file_hash(ip),
            environment_seed=es,
            policy_seed=ps,
            actions_sha256=action.hexdigest(),
            steps=steps,
            success=win,
        )
        tmp = f.with_suffix(".pending")
        tmp.write_text(json.dumps(row) + "\n")
        tmp.rename(f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("--shard", type=int, required=True)
    main(p.parse_args())
