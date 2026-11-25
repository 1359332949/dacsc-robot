"""Prospective image-corruption transfer screen, not official LIBERO-Plus held-out.
New image configurations; lifetime freshness of source base states is not claimed.
"""

import argparse, ctypes, hashlib, json, os, random, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/hy-tmp/patch-diag")
LIB = Path("/hy-tmp/LIBERO-plus")
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "scripts"),
    str(ROOT / "repos/CoRP"),
    str(ROOT / "repos/CoRP/scripts"),
    str(LIB),
]
from dacsc.static_visual_v2 import (
    array_hash,
    file_hash,
    seed_for,
    sensor_transform,
    append_event,
)
from dacsc.single_block_probe import align, VIEWS
from egl_shallow_block3_closed_loop import match_pixels, CHECKPOINT, KEYS
from block3_object_clean_compat import OBJECTS

ARMS = ("frozen", "spatial_only", "object_calibrated")
FACTORS = ("sensor_mix", "photometric", "image_texture_overlay")
TAG = "object-transfer-synthetic-v1"


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def params(task, j, factor, tag=TAG):
    g = np.random.default_rng(seed_for(tag, task, j, factor, "configuration"))
    if factor == "clean":
        return {}
    if factor == "sensor_mix":
        return dict(code=(3, 15, 27)[j % 3], mix=float(g.uniform(0.70, 0.95)))
    if factor == "photometric":
        return dict(
            exposure=float(g.uniform(0.65, 1.35)),
            gamma=float(g.uniform(0.75, 1.35)),
            contrast=float(g.uniform(0.75, 1.25)),
            red_blue=float(g.uniform(-0.15, 0.15)),
        )
    return dict(
        opacity=float(g.uniform(0.08, 0.18)),
        coarse_cells=[8, 32],
        field_seed=seed_for(tag, task, j, "texture_field"),
    )


def modify(obs, c, step, wrapper):
    if c["factor"] == "clean":
        return obs
    out = dict(obs)
    p = c["params"]
    factor = c["factor"]
    for i, k in enumerate(KEYS):
        if factor == "sensor_mix" and c["presentation"] == "standard" and i == 1:
            continue
        x = np.asarray(obs[k])
        assert x.dtype == np.uint8 and x.ndim == 3 and x.shape[-1] == 3
        ns = c["sensor_seed"] if i == 0 else seed_for(c["sensor_seed"], "wrist")
        if factor == "sensor_mix":
            y = sensor_transform(
                dict(obs, agentview_image=x), p["code"], ns, step, wrapper
            )["agentview_image"]
            v = (1 - p["mix"]) * x.astype(float) + p["mix"] * np.asarray(y).astype(
                float
            )
        elif factor == "photometric":
            y = x.astype(float) / 255.0
            y = np.maximum(0, (y - 0.5) * p["contrast"] + 0.5)
            v = (
                255
                * p["exposure"]
                * np.power(y, p["gamma"])
                * np.exp(np.array([p["red_blue"], 0.0, -p["red_blue"]]))
            )
        else:
            h, w = x.shape[:2]
            g = np.random.default_rng(seed_for(p["field_seed"], i))
            field = np.zeros_like(x, dtype=float)
            for cells in p["coarse_cells"]:
                z = g.integers(0, 256, (cells, cells, 3), dtype=np.uint8)
                field += np.asarray(
                    Image.fromarray(z).resize((w, h), Image.Resampling.BILINEAR),
                    dtype=float,
                ) / len(p["coarse_cells"])
            v = (1 - p["opacity"]) * x.astype(float) + p["opacity"] * field
        out[k] = np.clip(np.rint(v), 0, 255).astype(np.uint8)
    return out


def plan(a):
    import torch

    assert not (a.root / "manifest.json").exists()
    a.root.mkdir(exist_ok=True, parents=True)
    smoke = json.loads((a.root / "object_clean_smoke.json").read_text())["rows"]
    source = json.loads((a.root / "source_verification.json").read_text())
    assert len(source["rows"]) == 10
    ban = {v["raw_init_hash"] for v in smoke + source["rows"]}
    settled = {v["settled_state_sha256"] for v in smoke + source["rows"]}
    jobs = []
    pre = []

    def case(task, idx, j, factor, pres, tag):
        init = LIB / "libero/libero/init_files/libero_object" / (task + ".pruned_init")
        bd = LIB / "libero/libero/bddl_files/libero_object" / (task + ".bddl")
        st = torch.load(init, weights_only=False, map_location="cpu")[idx]
        raw = array_hash(np.asarray(st))
        assert raw not in ban
        c = dict(
            task=task,
            init_index=idx,
            raw_init_hash=raw,
            init_file=str(init),
            init_file_sha256=file_hash(init),
            bddl=str(bd),
            bddl_sha256=file_hash(bd),
            factor=factor,
            presentation=pres,
            params=params(task, j, factor, tag),
            env_seed=seed_for(tag, task, idx, "env"),
            sensor_seed=seed_for(tag, task, j, "sensor"),
            augmentation_seed=seed_for(tag, task, j, "augmentation"),
        )
        c["id"] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[
            :24
        ]
        return c

    for ob in OBJECTS:
        task = f"pick_up_the_{ob}_and_place_it_in_the_basket"
        jobs.append(case(task, 6, 0, "clean", "standard", TAG))
        for j in range(3):
            for f in FACTORS:
                for pr in (
                    ("standard", "bilateral") if f == "sensor_mix" else ("standard",)
                ):
                    jobs.append(case(task, 3 + j, j, f, pr, TAG))
    assert len(jobs) == 130 and len({v["id"] for v in jobs}) == 130
    task = jobs[0]["task"]
    for f in ("clean", *FACTORS):
        for pr in ("standard", "bilateral") if f == "sensor_mix" else ("standard",):
            pre.append(case(task, 7, 1, f, pr, TAG + "-preflight"))
    assert len(pre) == 5 and not {r["raw_init_hash"] for r in pre} & {
        r["raw_init_hash"] for r in jobs
    }
    artifact = {
        k: file_hash(a.root / k)
        for k in (
            "spatial_transfer.npz",
            "object_loto.npz",
            "source_verification.json",
            "object_clean_smoke.json",
        )
    }
    assert (
        artifact["spatial_transfer.npz"]
        == "8cffb42f96d0275fe7277631825ec0f96c6471acb87a2692125e80088dc15bd9"
        and artifact["object_loto.npz"] == source["ref_sha256"]
    )
    files = {
        str(p): file_hash(p)
        for p in [
            Path(__file__).resolve(),
            Path(__file__).resolve().with_name("report_object_transfer_v1.py"),
            ROOT / "dacsc/single_block_probe.py",
            ROOT / "dacsc/static_visual_v2.py",
            ROOT / "scripts/target_only_action_validation.py",
            ROOT / "scripts/egl_shallow_block3_closed_loop.py",
            ROOT / "repos/CoRP/scripts/evaluate_libero.py",
            LIB / "libero/libero/envs/env_wrapper.py",
        ]
    }
    lock = json.loads((a.root / "method_lock.json").read_text())
    assert (
        file_hash(ROOT / "dacsc/single_block_probe.py")
        == lock["files"]["dacsc/single_block_probe.py"]
    )
    assert (
        file_hash(LIB / "libero/libero/envs/env_wrapper.py")
        == lock["server_wrapper_sha256"]
    )
    artifact["method_lock.json"] = file_hash(a.root / "method_lock.json")
    artifact["PROTOCOL.md"] = file_hash(a.root / "PROTOCOL.md")
    write(
        a.root / "manifest.json",
        dict(
            kind="prospective_synthetic_transfer",
            role="frozen_candidate_one_shot",
            no_lifetime_state_freshness_claim=True,
            jobs=jobs,
            preflight=pre,
            policy_seeds=[0, 1, 2],
            banned_raw=sorted(ban),
            banned_settled=sorted(settled),
            artifacts=artifact,
            code=files,
            checkpoint=CHECKPOINT,
        ),
    )
    (a.root / "manifest.sha256").write_text(file_hash(a.root / "manifest.json"))


def load_plan(root):
    h = file_hash(root / "manifest.json")
    assert h == (root / "manifest.sha256").read_text()
    m = json.loads((root / "manifest.json").read_text())
    assert (
        m["kind"] == "prospective_synthetic_transfer"
        and m["role"] == "frozen_candidate_one_shot"
    )
    for k, v in m["artifacts"].items():
        assert file_hash(root / k) == v, ("artifact changed", k)
    for k, v in m["code"].items():
        assert file_hash(Path(k)) == v, ("code changed", k)
    return m, h


def run(a):
    import torch
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

    assert os.environ["MUJOCO_GL"] == os.environ["PYOPENGL_PLATFORM"] == "egl"
    m, mh = load_plan(a.root)
    cp = ROOT / "repos/CoRP/checkpoints/corp_libero.pt"
    assert file_hash(cp) == CHECKPOINT
    assert SETTLING_STEPS == 10
    if not a.preflight:
        permit = json.loads((a.root / "preflight_pass.json").read_text())
        assert permit["manifest"] == mh and permit["passed"]
        assert (a.root / "release.sha256").read_text() == mh
    model = load_policy(cp, device="cuda:0").eval().requires_grad_(False)
    ext = model.representation_extractor
    assert (
        tuple(ext.view_names) == VIEWS and len(ext.visual_encoder.backbone.blocks) == 12
    )
    with np.load(a.root / "spatial_transfer.npz", allow_pickle=False) as z:
        sp = tuple(
            torch.as_tensor(z[k], device="cuda:0") for k in ("mean", "std", "tau")
        )
    with np.load(a.root / "object_loto.npz", allow_pickle=False) as z:
        refs = {
            t: tuple(
                torch.as_tensor(z[t + "__3__" + k], device="cuda:0")
                for k in ("mean", "std", "tau")
            )
            for t in {c["task"] for c in m["jobs"]}
        }
    directory = a.root / ("preflight" if a.preflight else "trials")
    directory.mkdir(exist_ok=True)
    claims = a.root / ("preflight_claims" if a.preflight else "claims")
    claims.mkdir(exist_ok=True)
    items = [
        (s, c)
        for s in ([0] if a.preflight else m["policy_seeds"])
        for c in (m["preflight"] if a.preflight else m["jobs"])
    ]
    for s, c in items:
        dest = directory / f"{s}_{c['id']}.json"
        if dest.exists():
            continue
        try:
            (claims / dest.stem).mkdir()
        except FileExistsError:
            continue
        assert c["raw_init_hash"] not in m["banned_raw"] and c["init_index"] not in (
            2,
            8,
            9,
        )
        assert (
            file_hash(Path(c["init_file"])) == c["init_file_sha256"]
            and file_hash(Path(c["bddl"])) == c["bddl_sha256"]
        )
        init = torch.load(c["init_file"], weights_only=False, map_location="cpu")[
            c["init_index"]
        ]
        assert array_hash(np.asarray(init)) == c["raw_init_hash"]
        append_event(
            a.root / "access.jsonl",
            dict(
                event="begin",
                manifest=mh,
                id=c["id"],
                policy_seed=s,
                preflight=a.preflight,
            ),
        )
        cache = {}
        rows = []
        lang = grab_language_from_filename("libero_object", c["task"] + ".bddl")
        ps = seed_for(TAG, c["task"], c["init_index"], s, "policy")
        for name in ARMS:
            start = time.monotonic()
            es = c["env_seed"]
            random.seed(es)
            np.random.seed(es)
            torch.manual_seed(es)
            ctypes.CDLL(None).srand(ctypes.c_uint(es))
            env = OffScreenRenderEnv(
                bddl_file_name=c["bddl"],
                camera_heights=256,
                camera_widths=256,
                render_gpu_device_id=0,
            )
            handle = None
            calls = 0
            trace = []
            audit = []
            ha = hashlib.sha256()
            ref = refs[c["task"]] if name == "object_calibrated" else sp

            def hook(module, args, x):
                nonlocal calls
                assert x.shape[0] == 2
                calls += 1
                if name == "frozen":
                    assert align(x, ref, 0.0, "dead") is x
                    return None
                return align(x, ref, 0.75, "dead")

            try:
                env.noise = 0
                env.seed(es)
                env.reset()
                obs = env.set_init_state(init)
                for _ in range(SETTLING_STEPS):
                    obs = env.step(DUMMY_ACTION)[0]
                initial_state = array_hash(env.sim.get_state().flatten())
                assert initial_state not in m["banned_settled"]
                handle = ext.visual_encoder.backbone.blocks[2].register_forward_hook(
                    hook
                )
                gen = torch.Generator(device="cuda:0").manual_seed(ps)
                steps = 0
                with torch.no_grad():
                    while steps < 220:
                        physical = array_hash(env.sim.get_state().flatten())
                        key = (
                            steps,
                            physical,
                            *(
                                array_hash(np.asarray(obs[k]))
                                for k in (
                                    "robot0_eef_pos",
                                    "robot0_eef_quat",
                                    "robot0_gripper_qpos",
                                )
                            ),
                        )
                        raw = match_pixels(obs, key, cache, audit)
                        obs2 = modify(raw, c, steps, wrapper)
                        assert array_hash(env.sim.get_state().flatten()) == physical
                        assert all(
                            np.array_equal(obs2[k], raw[k])
                            for k in (
                                "robot0_eef_pos",
                                "robot0_eef_quat",
                                "robot0_gripper_qpos",
                            )
                        )
                        z = torch.randn((1, 12, 7), device="cuda:0", generator=gen)
                        _, rep = visual_from_obs(
                            model, policy_observation(obs2, lang, model)
                        )
                        plan = (
                            model.normalizer.unnormalize_action(
                                sample_z(model, rep, z)
                            )[0]
                            .clamp(-1, 1)
                            .cpu()
                            .numpy()
                        )
                        assert plan.shape == (12, 7) and np.isfinite(plan).all()
                        trace.append(
                            dict(
                                step=steps,
                                state=physical,
                                raw=[array_hash(np.asarray(raw[k])) for k in KEYS],
                                images=[array_hash(np.asarray(obs2[k])) for k in KEYS],
                                z=array_hash(z.cpu().numpy()),
                                plan=array_hash(plan),
                            )
                        )
                        for ac in plan:
                            ha.update(ac.tobytes())
                            obs, rew, done, info = env.step(ac)
                            steps += 1
                            success = bool(is_success(env, info, rew, done))
                            if done or success or steps >= 220:
                                break
                        if done or success:
                            break
                assert calls == len(trace)
                if rows:
                    first = rows[0]["trace"][0]
                    assert all(
                        trace[0][k] == first[k] for k in ("state", "raw", "images", "z")
                    )
                    assert all(
                        v["z"] == b["z"] for v, b in zip(trace, rows[0]["trace"])
                    )
                rows.append(
                    dict(
                        method=name,
                        success=success,
                        steps=steps,
                        chunks=calls,
                        trace=trace,
                        roundings=audit,
                        action_sha=ha.hexdigest(),
                        seconds=time.monotonic() - start,
                    )
                )
            finally:
                if handle:
                    handle.remove()
                env.close()
        record = dict(
            id=c["id"],
            seed=s,
            task=c["task"],
            manifest=mh,
            factor=c["factor"],
            presentation=c["presentation"],
            policy_seed=ps,
            preflight=a.preflight,
            arms=rows,
        )
        tmp = dest.with_suffix(".pending")
        write(tmp, record)
        tmp.rename(dest)
        append_event(
            a.root / "access.jsonl",
            dict(
                event="end",
                id=c["id"],
                policy_seed=s,
                result=file_hash(dest),
                preflight=a.preflight,
            ),
        )
        print(dest.name, flush=True)


def admit(a):
    m, h = load_plan(a.root)
    fs = list((a.root / "preflight").glob("*.json"))
    assert len(fs) == 5
    ids = set()
    for p in fs:
        r = json.loads(p.read_text())
        assert (
            r["preflight"]
            and r["manifest"] == h
            and r["id"] in {c["id"] for c in m["preflight"]}
        )
        ids.add(r["id"])
        assert [v["method"] for v in r["arms"]] == list(ARMS)
        for v in r["arms"]:
            assert v["trace"] and v["chunks"] == len(v["trace"])
    assert len(ids) == 5
    write(
        a.root / "preflight_pass.json",
        dict(
            passed=True,
            manifest=h,
            n_cases=5,
            decision="integrity_only_no_success_selection",
        ),
    )
    (a.root / "release.sha256").write_text(h)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("plan", "run", "admit"))
    p.add_argument("root", type=Path)
    p.add_argument("--preflight", action="store_true")
    a = p.parse_args()
    {"plan": plan, "run": run, "admit": admit}[a.mode](a)
