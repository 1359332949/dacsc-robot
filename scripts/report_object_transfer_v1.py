"""Audit complete paired synthetic image-corruption trial grid, independent of runner."""

import argparse, hashlib, json
from pathlib import Path
from collections import Counter
import numpy as np

ARMS = ("frozen", "spatial_only", "object_calibrated")
GROUPS = (
    "clean",
    "sensor_single",
    "sensor_bilateral",
    "photometric",
    "image_texture_overlay",
    "standard100",
    "combined130",
)


def load(f):
    return json.loads(f.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summary(rows, method, ref):
    tasks = sorted({r["task"] for r in rows})
    x = np.array(
        [
            [
                sum(r["task"] == t for r in rows),
                sum(r["wins"][method] for r in rows if r["task"] == t),
                sum(r["wins"][ref] for r in rows if r["task"] == t),
            ]
            for t in tasks
        ]
    )
    w = np.random.default_rng(20261125).multinomial(
        len(tasks), np.full(len(tasks), 1 / len(tasks)), size=30000
    )
    b = w @ x
    ci = lambda x: np.quantile(x, [0.025, 0.975]).round(4).tolist()
    n, a, c = map(int, x.sum(0))
    return dict(
        n=n,
        tasks=len(tasks),
        wins=a,
        reference_wins=c,
        sr_pct=100 * a / n,
        delta_pp=100 * (a - c) / n,
        recovered=sum(r["wins"][method] > r["wins"][ref] for r in rows),
        harmed=sum(r["wins"][method] < r["wins"][ref] for r in rows),
        net=a - c,
        delta_ci=ci(100 * (b[:, 1] - b[:, 2]) / b[:, 0]),
        sr_ci=ci(100 * b[:, 1] / b[:, 0]),
    )


def main(a):
    root = a.root
    m = load(root / "manifest.json")
    mh = sha(root / "manifest.json")
    assert (
        mh == (root / "manifest.sha256").read_text()
        and len(m["jobs"]) == 130
        and m["kind"] == "prospective_synthetic_transfer"
    )
    assert (root / "release.sha256").read_text() == mh
    cases = {r["id"]: r for r in m["jobs"]}
    assert len(cases) == 130 and len(m["preflight"]) == 5
    assert not {r["raw_init_hash"] for r in m["preflight"]} & {
        r["raw_init_hash"] for r in m["jobs"]
    }
    for k, v in m["artifacts"].items():
        assert sha(root / k) == v
    for p, h in m["code"].items():
        q = Path(p)
        # Server paths are resolved against backed-up exact code when analyzing locally.
        if not q.exists():
            q = root / "code" / Path(p).name
            if not q.exists():
                q = Path("/private/tmp/patch-diag") / Path(p).relative_to(
                    "/hy-tmp/patch-diag"
                )
        assert sha(q) == h, ("code differs", p)
    source = load(root / "source_verification.json")
    smoke = load(root / "object_clean_smoke.json")["rows"]
    calib = source["rows"]
    assert len(calib) == 10 and len(smoke) == 20
    assert source["ref_sha256"] == sha(root / "object_loto.npz")
    assert all(
        t not in v["tasks"] and len(v["tasks"]) == 9 for t, v in source["folds"].items()
    )
    ban = {v["raw_init_hash"] for v in smoke + calib}
    assert set(m["banned_raw"]) == ban
    seen = set()
    rows = []
    first = {}
    roundings = Counter()
    h = hashlib.sha256()
    conditions = Counter()
    completed = set()
    events = [json.loads(l) for l in (root / "access.jsonl").read_text().splitlines()]
    for e in events:
        if e["event"] == "end" and not e["preflight"]:
            completed.add((e["id"], e["policy_seed"], e["result"]))
    for p in sorted((root / "trials").glob("*.json")):
        r = load(p)
        cid = r["id"]
        s = r["seed"]
        key = (cid, s)
        assert (
            key not in seen
            and cid in cases
            and s in (0, 1, 2)
            and not r["preflight"]
            and r["manifest"] == mh
        )
        seen.add(key)
        c = cases[cid]
        assert c["init_index"] not in (2, 7, 8, 9) and c["raw_init_hash"] not in ban
        assert (
            r["task"] == c["task"]
            and r["factor"] == c["factor"]
            and r["presentation"] == c["presentation"]
        )
        assert [v["method"] for v in r["arms"]] == list(ARMS)
        assert (cid, s, sha(p)) in completed
        h.update(p.name.encode() + bytes.fromhex(sha(p)))
        base = r["arms"][0]["trace"]
        assert base and base[0]["state"] not in m["banned_settled"]
        first[key] = base[0]
        wins = {}
        conditions[r["factor"], r["presentation"]] += 1
        for arm in r["arms"]:
            t = arm["trace"]
            assert (
                t
                and arm["chunks"] == len(t)
                and [z["step"] for z in t] == list(range(0, arm["steps"], 12))
                and 0 < arm["steps"] <= 220
                and type(arm["success"]) == bool
            )
            assert all(
                t[0][k] == base[0][k] for k in ("state", "raw", "images", "z")
            ) and all(v["z"] == b["z"] for v, b in zip(t, base))
            assert all(v[2] <= 64 and v[3] <= 1 for v in arm["roundings"])
            roundings[arm["method"]] += len(arm["roundings"])
            wins[arm["method"]] = int(arm["success"])
        groups = ["combined130"] + (
            ["sensor_bilateral"]
            if r["presentation"] == "bilateral"
            else [
                "standard100",
                "sensor_single" if c["factor"] == "sensor_mix" else c["factor"],
            ]
        )
        rows.append(dict(id=cid, seed=s, task=r["task"], groups=groups, wins=wins))
    assert seen == {(cid, s) for cid in cases for s in (0, 1, 2)}, (
        "missing groups",
        390 - len(seen),
    )
    assert len(rows) == 390
    cmp = Counter()
    for cid, c in cases.items():
        f = [first[cid, s] for s in (0, 1, 2)]
        assert len({v["state"] for v in f}) == 1 and len({v["z"] for v in f}) == 3
        if c["factor"] == "sensor_mix" and c["presentation"] == "bilateral":
            same = [
                i
                for i, v in cases.items()
                if v["factor"] == "sensor_mix"
                and v["task"] == c["task"]
                and v["init_index"] == c["init_index"]
                and v["presentation"] == "standard"
            ]
            assert len(same) == 1
            for s in (0, 1, 2):
                x, y = first[cid, s], first[same[0], s]
                assert all(x[k] == y[k] for k in ("state", "raw", "z"))
                assert (
                    x["images"][0] == y["images"][0]
                    and x["images"][1] != y["images"][1]
                )
                cmp["matching_single_bilateral"] += 1
    grid = {}
    for s in ("all", 0, 1, 2):
        grid[str(s)] = {}
        for g in GROUPS:
            sub = [
                r for r in rows if g in r["groups"] and (s == "all" or r["seed"] == s)
            ]
            grid[str(s)][g] = {
                k + "_vs_" + ref: summary(sub, k, ref)
                for k, ref in [(v, "frozen") for v in ARMS]
                + [("object_calibrated", "spatial_only")]
            }
    res = dict(
        status="COMPLETE_SYNTHETIC_SCREEN",
        manifest_sha256=mh,
        n_conditions=390,
        n_rollouts=1170,
        grid=grid,
        roundings=dict(roundings),
        source_chunks=source["chunks"],
        paired_verification=dict(cmp),
        rows=rows,
        trial_hash_chain=h.hexdigest(),
        limitations=[
            "New configurations of image-space corruptions, not official Lighting/Background assets",
            "Source base-state lifetime historical freshness unresolved",
            "Only10 tasks; state6 Clean has one start/task and three policy seeds",
            "Combined130 repeats30 Sensor identities",
            "Primary zero-target-calibration arm is fixed spatial_only; do not select arm from held-out outcomes",
        ],
    )
    a.out.mkdir(exist_ok=True, parents=True)
    (a.out / "summary.json").write_text(
        json.dumps(res, separators=(",", ":"), allow_nan=False) + "\n"
    )
    report(res, a.out / "OBJECT_TRANSFER_SYNTHETIC_V1.md")
    print("verified all390 triples,1170 actual rollouts")


def report(s, p):
    out = [
        "# Object 跨套件三臂：新生成的图像扰动 V1",
        "",
        "一次封存的新图像配置，不是官方 LIBERO-Plus 留出。主迁移臂 spatial_only 使用固定 Spatial 参考；object_calibrated 是独立列出的允许新域校准组。算法与 alpha=.75 均不按本次结果修改。光色变换和全画面纹理覆盖不冒充物理光照/背景。",
        "",
    ]
    for seed, gg in s["grid"].items():
        out += [
            f"## 种子 {seed}",
            "",
            "每格：成功数/分母；相对同期 Frozen 救回/误伤；成功率差 pp [任务聚类95%区间]。",
            "",
            "|条件|Frozen|Spatial参考|Object留一校准|校准相对Spatial参考|",
            "|---|---:|---:|---:|---:|",
        ]
        for g, v in gg.items():
            cols = []
            for k in (
                "frozen_vs_frozen",
                "spatial_only_vs_frozen",
                "object_calibrated_vs_frozen",
                "object_calibrated_vs_spatial_only",
            ):
                a = v[k]
                cols.append(
                    f"{a['wins']}/{a['n']}"
                    + (
                        f"；{a['recovered']}/{a['harmed']}；{a['delta_pp']:+.1f} [{a['delta_ci'][0]:.1f},{a['delta_ci'][1]:.1f}]"
                        if k != "frozen_vs_frozen"
                        else ""
                    )
                )
            out.append("|" + g + "|" + "|".join(cols) + "|")
        out += [""]
    out += [
        "所有区间按基础任务成簇重采样30,000次，同一任务所有实例、相机条件、种子一起抽。仅10任务，不能推出跨策略或跨机器人普适性；Clean恒等胜负也不推出零风险。双眼条件复用标准Sensor身份，combined130仅描述。原始状态是否历史上从未接触无法证明；该批新扰动配置是在冻结V1后封存。这里只作有限预算的前瞻迁移screen，不事后挑最好种子、阈值、域或参考。",
        "",
    ]
    p.write_text("\n".join(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    main(p.parse_args())
