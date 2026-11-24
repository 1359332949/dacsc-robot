"""Read pinned rollout archive; replay *only* the selected Dev contrasts to recover absent telemetry."""

import argparse, ctypes, hashlib, json, os, random, sys, tarfile
from collections import Counter
from pathlib import Path
import numpy as np

ARCHIVE = "02de5f0696d2213176ed861a14c462fd19813e8cae226377cc08a26ee608c33d"


def select(archive):
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE
    with tarfile.open(archive) as t:
        data = {
            p.name: json.load(t.extractfile(p))
            for p in t
            if p.name.startswith("trials/") and p.name.endswith(".json")
        }
    assert len(data) == 390
    selected = []
    controls = []
    for name, r in sorted(data.items()):
        if r["regime"] != "standard":
            continue
        w = [a["success"] for a in r["arms"]]
        g = None
        if r["factor"] == "background" and r["sub"] == "table_texture" and w[0] != w[2]:
            g = "table_harm" if w[0] else "table_rescue"
        if r["factor"] == "clean":
            if w[0] and not w[2]:
                g = "clean_harm"
            if w[0] and w[2]:
                controls.append(r)
        if g:
            selected.append((g, r))
    tasks = {r["task"] for g, r in selected if g == "clean_harm"}
    for s in range(3):
        r = next(r for r in controls if r["seed"] == s and r["task"] not in tasks)
        tasks.add(r["task"])
        selected.append(("clean_control", r))
    assert Counter(g for g, r in selected) == {
        "table_harm": 7,
        "table_rescue": 2,
        "clean_harm": 1,
        "clean_control": 3,
    }
    return selected


def replay(a, picks):
    import torch
    import egl_shallow_block3_closed_loop as src
    from dacsc.single_block_probe import score, align, VIEWS
    from dacsc.static_visual_v2 import array_hash, file_hash, authorize
    from corp import load_policy
    from evaluate_libero import (
        DUMMY_ACTION,
        SETTLING_STEPS,
        is_success,
        policy_observation,
    )
    from target_only_action_validation import sample_z, visual_from_obs
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.benchmark import grab_language_from_filename
    import libero.libero.envs.env_wrapper as wrapper

    assert (
        0 <= a.shard < a.workers
        and os.environ["MUJOCO_GL"] == os.environ["PYOPENGL_PLATFORM"] == "egl"
    )
    mp = src.ROOT / "docs/data/static_visual_v2_suite/final/dev_manifest.json"
    assert file_hash(mp) == src.MANIFEST
    m = json.loads(mp.read_text())
    assert m["kind"] == "dev"
    cases = {c["case_id"]: c for c in m["cases"]}
    refdir = src.ROOT / "runs/invariance_sensitivity_v1b"
    assert (
        file_hash(refdir / "references.npz") == src.REFERENCE
        and file_hash(refdir / "sources.json") == src.SOURCE
    )
    folds = json.loads((refdir / "sources.json").read_text())
    cp = src.CORP / "checkpoints/corp_libero.pt"
    assert file_hash(cp) == src.CHECKPOINT
    model = load_policy(cp, device="cuda:0").eval().requires_grad_(False)
    ext = model.representation_extractor
    assert tuple(ext.view_names) == VIEWS
    a.out.mkdir(parents=True, exist_ok=True)
    for group, r in picks[a.shard :: a.workers]:
        c = cases[r["case_id"]]
        task = r["task"]
        assert task == c["base_task"] and r["reference"] == src.REFERENCE
        assert set(folds[task]) == {
            v["case_id"]
            for v in cases.values()
            if v["factor"] == "clean" and v["base_task"] != task
        }
        authorize(m, c, "table harm selected Dev replay", src.MANIFEST, mp)
        dest = a.out / f"{r['case_id']}_{r['seed']}.json"
        assert not dest.exists()
        with np.load(refdir / "references.npz", allow_pickle=False) as z:
            ref = tuple(
                torch.as_tensor(z[f"{task}__3__{k}"], device="cuda:0")
                for k in ("mean", "std", "tau")
            )
        assert (
            file_hash(src.LIB / c["bddl"]) == c["resolved_bddl_hash"]
            and file_hash(src.LIB / c["init_state_file"]) == c["init_state_file_sha256"]
        )
        init = torch.load(
            src.LIB / c["init_state_file"], weights_only=False, map_location="cpu"
        )[c["init_state_index"]]
        assert array_hash(np.asarray(init)) == c["raw_init_hash"]
        lang = grab_language_from_filename(c["suite"], task + ".bddl")
        assert not c["runtime_params"]["noise"]
        cache = {}
        result = {}
        for arc in [r["arms"][0]] if a.frozen_only else [r["arms"][0], r["arms"][2]]:
            name = arc["method"]
            assert name in ("frozen", "dead075")
            es = r["env_seed"]
            random.seed(es)
            np.random.seed(es)
            torch.manual_seed(es)
            ctypes.CDLL(None).srand(ctypes.c_uint(es))
            env = OffScreenRenderEnv(
                bddl_file_name=str(src.LIB / c["bddl"]),
                camera_heights=256,
                camera_widths=256,
                render_gpu_device_id=0,
            )
            trace = []
            probe = []
            audit = []
            handle = None

            def hook(mod, args, x):
                patch = x[:, 1:]
                mt = patch.mean(1)
                st = patch.std(1, unbiased=False)
                ms, ss, tau = ref
                active = score(mt, st, ms, ss) > tau
                ratio = ((0.25 * st + 0.75 * ss) / (st.square() + 1e-6).sqrt()).clamp(
                    0.5, 2
                )
                g = torch.where(active, ratio, 1.0).sub(1).abs().mean(1)
                probe.append(
                    {"active": active.float().mean(1).tolist(), "scale": g.tolist()}
                )
                return align(x, ref, 0.75, "dead") if name == "dead075" else None

            try:
                env.noise = 0
                env.seed(es)
                env.reset()
                obs = env.set_init_state(init)
                for _ in range(SETTLING_STEPS):
                    obs = env.step(DUMMY_ACTION)[0]
                handle = ext.visual_encoder.backbone.blocks[2].register_forward_hook(
                    hook
                )
                gen = torch.Generator(device="cuda:0").manual_seed(r["policy_seed"])
                steps = 0
                h = hashlib.sha256()
                body = env.sim.model.body_name2id(
                    env.env.objects_dict["akita_black_bowl_1"].root_body
                )
                with torch.no_grad():
                    while steps < 220:
                        raw = src.observed(
                            obs, 0, r["sensor_seed"], steps, False, wrapper
                        )
                        key = (
                            steps,
                            array_hash(env.sim.get_state().flatten()),
                            *(
                                array_hash(np.asarray(raw[k]))
                                for k in (
                                    "robot0_eef_pos",
                                    "robot0_eef_quat",
                                    "robot0_gripper_qpos",
                                )
                            ),
                        )
                        raw = src.match_pixels(raw, key, cache, audit)
                        po = policy_observation(raw, lang, model)
                        z = torch.randn((1, 12, 7), device="cuda:0", generator=gen)
                        _, rep = visual_from_obs(model, po)
                        plan = (
                            model.normalizer.unnormalize_action(
                                sample_z(model, rep, z)
                            )[0]
                            .clamp(-1, 1)
                            .cpu()
                            .numpy()
                        )
                        hashes = dict(
                            step=steps,
                            state=key[1],
                            images=[array_hash(np.asarray(raw[k])) for k in src.KEYS],
                            z=array_hash(z.cpu().numpy()),
                            plan=array_hash(plan),
                        )
                        j = len(trace)
                        expected = arc["trace"][j] if j < len(arc["trace"]) else {}
                        assert len(probe) == j + 1
                        trace.append(
                            dict(
                                verified=hashes == expected,
                                match={
                                    k: v == expected.get(k) for k, v in hashes.items()
                                },
                                plan=plan.tolist(),
                                eef=np.asarray(raw["robot0_eef_pos"]).tolist(),
                                bowl=env.sim.data.body_xpos[body].copy().tolist(),
                                **probe[-1],
                            )
                        )
                        for ac in plan:
                            h.update(ac.tobytes())
                            obs, rew, done, info = env.step(ac)
                            steps += 1
                            success = bool(is_success(env, info, rew, done))
                            if done or success or steps >= 220:
                                break
                        if done or success:
                            break
                result[name] = dict(
                    trace=trace,
                    steps=steps,
                    success=success,
                    full_match=bool(
                        all(v["verified"] for v in trace)
                        and h.hexdigest() == arc["action_sha"]
                        and steps == arc["steps"]
                        and success == arc["success"]
                        and len(trace) == len(arc["trace"])
                    ),
                )
            finally:
                if handle:
                    handle.remove()
                env.close()
        dest.write_text(
            json.dumps(
                dict(
                    group=group, id=r["case_id"], seed=r["seed"], task=task, arms=result
                ),
                allow_nan=False,
            )
        )
        print(dest.name, {k: v["full_match"] for k, v in result.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("select", "replay"))
    p.add_argument("archive", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--case", action="append")
    p.add_argument("--frozen-only", action="store_true")
    a = p.parse_args()
    picks = select(a.archive)
    if a.case:
        picks = [(g, r) for g, r in picks if f"{r['case_id']}_{r['seed']}" in a.case]
        assert len(picks) == len(set(a.case))
    if a.mode == "select":
        print("\n".join(f"{g} {r['case_id']} {r['seed']}" for g, r in picks))
    else:
        replay(a, picks)
