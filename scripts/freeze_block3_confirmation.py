"""Produce frozen spatial and transfer reference artifacts solely from archived Spatial Clean."""

import argparse, hashlib, io, json, tarfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from dacsc.single_block_probe import fit, score

ARCH = "9e2ef5ec51b51d4c8129cca1a14d47d206ffca07b1ec771b0dfa99e344b7828c"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def cross(raw, ts):
    stats = lambda keys: fit(torch.cat([raw[k] for k in keys]))
    mu, sd = stats(ts)
    s = []
    for t in ts:
        mi, si = stats([k for k in ts if k != t])
        m, q = raw[t].unbind(1)
        s.append(score(m, (q - m.square()).clamp_min(0).sqrt(), mi, si))
    return [v.float().numpy() for v in (mu, sd, torch.cat(s).quantile(0.95, dim=0))]


def main(a):
    assert sha(a.archive.read_bytes()) == ARCH
    a.out.mkdir(exist_ok=True, parents=True)
    with tarfile.open(a.archive) as tar:
        spatial = tar.extractfile("full.npz").read()
        assert (
            sha(spatial)
            == "f5aa659cd4b4fe7e02a512bb1f884ebf19a6711c87f3c62831ddca004b11b218"
        )
        raw = {}
        for f in tar.getmembers():
            if not (
                f.name.startswith("source/standard_0_clean_")
                and f.name.endswith(".json")
            ):
                continue
            r = json.load(tar.extractfile(f))
            assert r["factor"] == "clean" and r["seed"] == 0
            raw[r["task"]] = torch.from_numpy(
                np.load(
                    io.BytesIO(
                        tar.extractfile("source/" + r["case_id"] + ".npy").read()
                    ),
                    allow_pickle=False,
                )
            )
    assert len(raw) == 10 and sum(map(len, raw.values())) == 101
    ts = sorted(raw)
    with np.load(io.BytesIO(spatial)) as old:
        for t in ts:
            for n, v in zip(
                ("mean", "std", "tau"), cross(raw, [k for k in ts if k != t])
            ):
                np.testing.assert_allclose(old[f"{t}__3__{n}"], v, rtol=2e-6, atol=2e-7)
    (a.out / "spatial_loto.npz").write_bytes(spatial)
    np.savez_compressed(
        a.out / "transfer_spatial10.npz",
        **dict(zip(("mean", "std", "tau"), cross(raw, ts))),
    )
    files = [
        "dacsc/single_block_probe.py",
        "dacsc/static_visual_v2.py",
        "scripts/target_only_action_validation.py",
        "scripts/egl_shallow_block3_closed_loop.py",
        "scripts/egl_block3_full_traj.py",
        "scripts/report_egl_full_traj.py",
        "scripts/freeze_block3_confirmation.py",
        "docs/data/static_visual_v2_suite/final/dev_manifest.json",
    ]
    lock = dict(
        method="block3_full_traj_dead075",
        method_commit="8576589d8c1baaf3f48820125428d0f084c646fb",
        artifact_source=ARCH,
        checkpoint="f933ab03c1a4664f7a11690645cdd5fa81788a5b98db81913024f7a23326dfe6",
        libero_plus="4976dc30028e805ff8094b55501d532c48fec182",
        server_wrapper_sha256="d3d3ddfa861fbdece477004532155be93250f9785e6e580ab9618a9daf8d7c27",
        server_evaluate_libero_sha256="8ec144c71cfcd3c0cab9d1b505d432062424487970575a4ff83c2bd0613d066a",
        block=3,
        alpha=0.75,
        tau_quantile=0.95,
        ema=False,
        views=["agentview", "eye_in_hand"],
        H=12,
        E=12,
        max_steps=220,
        source_chunks={t: len(raw[t]) for t in ts},
        refs={p.name: sha(p.read_bytes()) for p in a.out.glob("*.npz")},
        files={n: sha((ROOT / n).read_bytes()) for n in files},
    )
    (a.out / "method_lock.json").write_text(json.dumps(lock, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("out", type=Path)
    main(p.parse_args())
