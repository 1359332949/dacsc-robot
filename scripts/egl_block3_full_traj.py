"""LOTO full-trajectory source fit and chunk-window paired Dev rollouts."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORP = ROOT / "repos/CoRP"
LIB = Path("/hy-tmp/LIBERO-plus")
sys.path[:0] = [str(ROOT), str(CORP), str(CORP / "scripts"), str(LIB)]
from dacsc.static_visual_v2 import (
    array_hash,
    authorize,
    file_hash,
    seed_for,
    sensor_transform,
)
from dacsc.single_block_probe import VIEWS, align

KEYS = ("agentview_image", "robot0_eye_in_hand_image")
from egl_shallow_block3_closed_loop import observed, match_pixels
from dacsc.single_block_probe import fit, moments, score
import torch

MANIFEST = "d3e0d8c642cd215c879d2f0174663ff5ec585cd5c6d6b531a213c26e9fb982c0"
CHECKPOINT = "f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6"
SOURCE = "fbb03322881190546872010a3a94684991a7eab2f9b3f95943bfdce5ceac9e27"
REFERENCE = "9dceecd7ac5d343b49f2c2c7808becbd50ec6ed90b8077b7bd54c57e66f0f44f"


def fit_full(out, clean):
    raw = {
        t: torch.from_numpy(
            np.load(out / "source" / (cid + ".npy"), allow_pickle=False)
        )
        for t, cid in clean.items()
    }
    assert len(raw) == 10 and all(
        x.ndim == 4 and x.shape[1:] == (2, 2, 384) and torch.isfinite(x).all()
        for x in raw.values()
    )
    refs = {}
    audit = {}
    for t in sorted(raw):
        tasks = sorted(set(raw) - {t})
        assert len(tasks) == 9

        def stats(keys):
            return fit(torch.cat([raw[k] for k in keys]))

        mu, sd = stats(tasks)
        scores = []
        for hold in tasks:
            inner = [k for k in tasks if k != hold]
            assert len(inner) == 8 and t not in inner
            mi, si = stats(inner)
            m, q = raw[hold].unbind(1)
            scores.append(score(m, (q - m.square()).clamp_min(0).sqrt(), mi, si))
        tau = torch.cat(scores).quantile(0.95, dim=0)
        audit[t] = dict(tasks=tasks, frames=sum(len(raw[k]) for k in tasks))
        for n, x in zip(("mean", "std", "tau"), (mu, sd, tau)):
            refs[f"{t}__3__{n}"] = x.float().numpy()
    np.savez_compressed(out / "full.npz", **refs)
    (out / "folds.json").write_text(json.dumps(audit, indent=2))
    (out / "full.sha256").write_text(file_hash(out / "full.npz"))


def main():
    import torch
    from corp import load_policy
    from target_only_action_validation import sample_z, visual_from_obs
    from evaluate_libero import (
        DUMMY_ACTION,
        SETTLING_STEPS,
        is_success,
        policy_observation,
    )
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.benchmark import grab_language_from_filename
    import libero.libero.envs.env_wrapper as wrapper

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("mode", choices=("source", "fit", "run"))
    ap.add_argument("--resume-complete", action="store_true")
    a = ap.parse_args()
    assert (
        os.environ["MUJOCO_GL"] == os.environ["PYOPENGL_PLATFORM"] == "egl"
        and 0 <= a.shard < a.workers
    )
    final = ROOT / "docs/data/static_visual_v2_suite/final"
    mp = final / "dev_manifest.json"
    assert file_hash(mp) == MANIFEST
    m = json.loads(mp.read_text())
    cases = sorted(m["cases"], key=lambda c: c["case_id"])
    assert m["kind"] == "dev" and len(cases) == 100
    for c in cases:
        authorize(m, c, "paired block 3 EGL rollout", MANIFEST, mp)
    inputs = ROOT / "runs/invariance_sensitivity_v1b"
    assert (
        file_hash(inputs / "sources.json") == SOURCE
        and file_hash(inputs / "references.npz") == REFERENCE
    )
    clean = {c["base_task"]: c["case_id"] for c in cases if c["factor"] == "clean"}
    folds = json.loads((inputs / "sources.json").read_text())
    assert len(clean) == 10 and set(folds) == set(clean)
    with np.load(inputs / "references.npz", allow_pickle=False) as z:
        refs = {
            t: tuple(
                torch.as_tensor(z[f"{t}__3__{p}"], device="cuda:0")
                for p in ("mean", "std", "tau")
            )
            for t in clean
        }
    for t in clean:
        assert (
            len(folds[t]) == 9
            and set(folds[t]) == set(clean.values()) - {clean[t]}
            and all(x.shape == (2, 384) and x.dtype == torch.float32 for x in refs[t])
        )
    if a.mode == "fit":
        fit_full(a.out, clean)
        return
    if a.mode == "run":
        full_sha = file_hash(a.out / "full.npz")
        assert full_sha == (a.out / "full.sha256").read_text()
        with np.load(a.out / "full.npz") as z:
            full = {
                t: tuple(
                    torch.as_tensor(z[f"{t}__3__{p}"], device="cuda:0")
                    for p in ("mean", "std", "tau")
                )
                for t in clean
            }
    cp = CORP / "checkpoints/corp_libero.pt"
    assert file_hash(cp) == CHECKPOINT
    policy = load_policy(cp, device="cuda:0").eval().requires_grad_(False)
    ext = policy.representation_extractor
    assert (
        tuple(ext.view_names) == VIEWS and len(ext.visual_encoder.backbone.blocks) == 12
    )
    jobs = [
        (c, b, s)
        for s in (0, 1, 2)
        for c in cases
        for b in ([False, True] if c["factor"] == "sensor_noise" else [False])
    ]
    assert len(jobs) == 390
    if a.mode == "source":
        jobs = [j for j in jobs if j[0]["factor"] == "clean" and j[2] == 0]
        assert len(jobs) == 10
    sub = a.out / ("source" if a.mode == "source" else "trials")
    sub.mkdir(parents=True, exist_ok=True)
    # Start expensive bilateral glass blur cases early; shared filesystem claims distribute remaining cases dynamically.
    if a.mode == "run":
        jobs.sort(
            key=lambda j: (
                -(j[0]["runtime_params"]["noise"] > 40) * (1 + j[1]),
                j[2],
                j[0]["case_id"],
                j[1],
            )
        )
        (a.out / "claims").mkdir(exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for c, b, s in jobs[a.shard :: a.workers] if a.mode == "source" else jobs:
        cid = c["case_id"]
        task = c["base_task"]
        regime = "dual" if b else "standard"
        dest = sub / f"{regime}_{s}_{cid}.json"
        if a.mode == "run":
            if dest.exists():
                continue
            try:
                (a.out / "claims" / dest.stem).mkdir()
            except FileExistsError:
                continue
        seeds = json.loads((final / "results" / (cid + ".json")).read_text())["seeds"]
        es = seeds["environment_seed"]
        ns = seeds["sensor_noise_seed"]
        ps = (
            seeds["policy_flow_seed"]
            if not s
            else seed_for(c["pair_group_id"], s, "egl_policy_seed")
        )
        lang = grab_language_from_filename(c["suite"], task + ".bddl")
        code = int(c["runtime_params"]["noise"])
        assert (
            file_hash(LIB / c["bddl"]) == c["resolved_bddl_hash"]
            and file_hash(LIB / c["init_state_file"]) == c["init_state_file_sha256"]
        )
        initial = torch.load(
            LIB / c["init_state_file"], weights_only=False, map_location="cpu"
        )[c["init_state_index"]]
        assert array_hash(np.asarray(initial)) == c["raw_init_hash"]
        assert not dest.exists()
        rows = []
        cache = {}
        names = (
            ["source"]
            if a.mode == "source"
            else ["frozen", "dead075", "full_traj_dead075"]
            + (
                ["early_0_2", "late_3_plus"]
                if not b
                and c["factor"] == "background"
                and c["sub_dimension"] == "table_texture"
                else []
            )
        )
        sketches = []
        for name in names:
            ref = full[task] if name == "full_traj_dead075" else refs[task]
            activation = []
            active_chunks = []
            start = time.monotonic()
            random.seed(es)
            np.random.seed(es)
            torch.manual_seed(es)
            ctypes.CDLL(None).srand(ctypes.c_uint(es))
            env = OffScreenRenderEnv(
                bddl_file_name=str(LIB / c["bddl"]),
                camera_heights=256,
                camera_widths=256,
                render_gpu_device_id=0,
            )
            hook = None
            calls = 0

            def fix(module, args, x):
                nonlocal calls
                assert x.shape[0] == 2
                chunk = calls
                calls += 1
                if name == "source":
                    sketches.append(
                        torch.stack(moments(x[:, 1:].double())).cpu().numpy()
                    )
                    return None
                enabled = (
                    name != "frozen"
                    and (name != "early_0_2" or chunk < 3)
                    and (name != "late_3_plus" or chunk >= 3)
                )
                if c["factor"] == "clean":
                    patch = x[:, 1:]
                    m = patch.mean(1)
                    sd = patch.std(1, unbiased=False)
                    activation.append(
                        {
                            k: (score(m, sd, rr[0], rr[1]) > rr[2])
                            .float()
                            .mean(1)
                            .tolist()
                            for k, rr in [("old", refs[task]), ("full", full[task])]
                        }
                    )
                if enabled:
                    active_chunks.append(chunk)
                    return align(x, ref, 0.75, "dead")
                return None

            try:
                env.noise = 0
                env.seed(es)
                env.reset()
                obs = env.set_init_state(initial)
                for _ in range(SETTLING_STEPS):
                    obs = env.step(DUMMY_ACTION)[0]
                hook = ext.visual_encoder.backbone.blocks[2].register_forward_hook(fix)
                g = torch.Generator(device="cuda:0").manual_seed(ps)
                steps = 0
                trace = []
                audit = []
                h = hashlib.sha256()
                with torch.no_grad():
                    while steps < 220:
                        raw = observed(obs, code, ns, steps, b, wrapper)
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
                        raw = match_pixels(raw, key, cache, audit)
                        po = policy_observation(raw, lang, policy)
                        z = torch.randn((1, 12, 7), device="cuda:0", generator=g)
                        _, rep = visual_from_obs(policy, po)
                        plan = (
                            policy.normalizer.unnormalize_action(
                                sample_z(policy, rep, z)
                            )[0]
                            .clamp(-1, 1)
                            .cpu()
                            .numpy()
                        )
                        assert plan.shape == (12, 7) and np.isfinite(plan).all()
                        trace.append(
                            dict(
                                step=steps,
                                state=key[1],
                                images=[array_hash(np.asarray(raw[k])) for k in KEYS],
                                z=array_hash(z.cpu().numpy()),
                                plan=array_hash(plan),
                            )
                        )
                        for action in plan:
                            h.update(action.tobytes())
                            obs, rew, done, info = env.step(action)
                            steps += 1
                            success = bool(is_success(env, info, rew, done))
                            if done or success or steps >= 220:
                                break
                        if done or success:
                            break
                assert calls == len(trace)
                if rows:
                    f = rows[0]["trace"]
                    assert all(trace[0][k] == f[0][k] for k in ("images", "state", "z"))
                    assert all(x["z"] == y["z"] for x, y in zip(f, trace))
                rows.append(
                    dict(
                        method=name,
                        success=success,
                        steps=steps,
                        chunks=len(trace),
                        hook_calls=calls,
                        active_chunks=active_chunks,
                        activation=activation,
                        trace=trace,
                        roundings=audit,
                        action_sha=h.hexdigest(),
                        seconds=time.monotonic() - start,
                    )
                )
            finally:
                if hook is not None:
                    hook.remove()
                env.close()
        if a.mode == "source":
            np.save(sub / (cid + ".npy"), np.stack(sketches))
            assert len(sketches) == len(rows[0]["trace"])
        tmp = dest.with_suffix(".pending")
        tmp.write_text(
            json.dumps(
                dict(
                    case_id=cid,
                    task=task,
                    regime=regime,
                    seed=s,
                    factor=c["factor"],
                    sub=c["sub_dimension"],
                    reference=REFERENCE,
                    full_reference=(full_sha if a.mode == "run" else None),
                    env_seed=es,
                    sensor_seed=ns,
                    policy_seed=ps,
                    arms=rows,
                ),
                allow_nan=False,
            )
        )
        tmp.rename(dest)
        print(dest.name, flush=True)


if __name__ == "__main__":
    main()
