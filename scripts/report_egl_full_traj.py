"""Independently verify the LOTO fits and paired rollouts before scoring."""

import argparse, hashlib, json, tarfile
from collections import Counter
from pathlib import Path
import numpy as np
from report_egl_block3_closed_loop import GROUPS, DOC

ARMS = ("frozen", "dead075", "full_traj_dead075")
WINDOWS = ("early_0_2", "late_3_plus")
OLD = "9dceecd7ac5d343b49f2c2c7808becbd50ec6ed90b8077b7bd54c57e66f0f44f"
ARCH = "02de5f0696d2213176ed861a14c462fd19813e8cae226377cc08a26ee608c33d"


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def load(f):
    return json.loads(f.read_text())


def verify_sources(root, prior):
    rows = [load(f) for f in sorted((root / "source").glob("*.json"))]
    assert len(rows) == 10
    raw = {}
    matched = Counter()
    for r in rows:
        assert r["seed"] == 0 and r["factor"] == "clean" and r["regime"] == "standard"
        arm = r["arms"][0]
        ar = prior[(0, r["case_id"], "standard")]["arms"][0]
        x = np.load(root / "source" / (r["case_id"] + ".npy"), allow_pickle=False)
        assert (
            len(arm["trace"]) == len(x) == arm["hook_calls"]
            and arm["active_chunks"] == []
            and np.isfinite(x).all()
        )
        raw[r["task"]] = x
        for k in ["success", "action_sha", "steps", "trace"]:
            matched[k] += ar[k] == arm[k]
    folds = load(root / "folds.json")
    assert (
        set(folds) == set(raw)
        and len(raw) == 10
        and sha(root / "full.npz") == (root / "full.sha256").read_text()
    )
    err = []
    qs = []
    with np.load(root / "full.npz") as z:
        for t, d in folds.items():
            tasks = sorted(set(raw) - {t})
            assert (
                tasks == d["tasks"]
                and len(tasks) == 9
                and d["frames"] == sum(len(raw[k]) for k in tasks)
            )

            def fit(ks):
                a = np.concatenate([raw[k] for k in ks])
                mu = a[:, 0].mean(0)
                return mu, np.sqrt(np.maximum(0, a[:, 1].mean(0) - mu * mu))

            scores = []
            mu, sd = fit(tasks)
            for hold in tasks:
                inn = [k for k in tasks if k != hold]
                assert len(inn) == 8 and t not in inn
                mi, si = fit(inn)
                m = raw[hold][:, 0]
                s = np.sqrt(np.maximum(0, raw[hold][:, 1] - m * m))
                scores.append(((m - mi) ** 2 + (s - si) ** 2) / (si * si + 1e-6))
            tau = np.quantile(np.concatenate(scores), 0.95, axis=0)
            qs.append(np.median(tau))
            for n, v in [("mean", mu), ("std", sd), ("tau", tau)]:
                val = z[f"{t}__3__{n}"]
                np.testing.assert_allclose(v, val, rtol=2e-6, atol=2e-7)
                err.append(float(np.max(np.abs(v - val))))
    return dict(
        n_tasks=len(raw),
        chunks={k: len(v) for k, v in raw.items()},
        n_chunks=sum(map(len, raw.values())),
        fold_frames=[v["frames"] for v in folds.values()],
        reference_sha256=sha(root / "full.npz"),
        max_np_vs_torch_abs_error=max(err),
        source_vs_archive=dict(matched),
    )


def stats(rows, method, ref):
    ts = sorted({r["task"] for r in rows})
    x = np.array(
        [
            [
                sum(v["task"] == t for v in rows),
                sum(v["wins"][method] for v in rows if v["task"] == t),
                sum(v["wins"][ref] for v in rows if v["task"] == t),
            ]
            for t in ts
        ]
    )
    w = np.random.default_rng(20261124).multinomial(
        len(ts), np.full(len(ts), 1 / len(ts)), size=30000
    )
    boot = w @ x
    ci = lambda a: np.quantile(a, [0.025, 0.975]).round(3).tolist()
    n, b, a = map(int, x.sum(0))
    return dict(
        n=n,
        tasks=len(ts),
        wins=b,
        reference_wins=a,
        sr_pct=round(100 * b / n, 3),
        recovered=sum(r["wins"][method] > r["wins"][ref] for r in rows),
        harmed=sum(r["wins"][method] < r["wins"][ref] for r in rows),
        net=b - a,
        delta_pp=round(100 * (b - a) / n, 3),
        sr_ci=ci(100 * boot[:, 1] / boot[:, 0]),
        delta_ci=ci(100 * (boot[:, 1] - boot[:, 2]) / boot[:, 0]),
    )


def main(a):
    root = a.root
    assert sha(a.previous) == ARCH
    with tarfile.open(a.previous) as tar:
        prior = {
            (r["seed"], r["case_id"], r["regime"]): r
            for f in tar
            if f.name.startswith("trials/") and f.name.endswith(".json")
            for r in [json.load(tar.extractfile(f))]
        }
        assert len(prior) == 390
    sources = verify_sources(root, prior)
    print("source check", sources, flush=True)
    if a.sources_only:
        return
    cases = {
        v["case_id"]: v
        for v in load(DOC / "static_visual_v2_suite/final/dev_manifest.json")["cases"]
    }
    diag = load(DOC / "block3_table_harm_v1/summary.json")["cases"]
    focus = {(v["seed"], v["id"]) for v in diag if v["group"].startswith("clean")}
    seven = {(v["seed"], v["id"]) for v in diag if v["group"] == "table_harm"}
    assert len(focus) == 4 and len(seven) == 7
    seen = set()
    rows = []
    tele = []
    inj = []
    archive = Counter()
    first = {}
    seeds = {}
    h = hashlib.sha256()
    n_rollouts = 0
    for path in sorted((root / "trials").glob("*.json")):
        r = load(path)
        s, cid, reg = r["seed"], r["case_id"], r["regime"]
        key = (s, cid, reg)
        c = cases[cid]
        assert key in prior and key not in seen
        seen.add(key)
        h.update(path.name.encode() + bytes.fromhex(sha(path)))
        assert (
            r["reference"] == OLD and r["full_reference"] == sources["reference_sha256"]
        )
        assert (c["factor"], c["base_task"], c["sub_dimension"]) == (
            r["factor"],
            r["task"],
            r["sub"],
        )
        groups = ["combined130"] + (
            ["dual_sensor"]
            if reg == "dual"
            else [
                "standard100",
                "single_sensor" if c["factor"] == "sensor_noise" else c["factor"],
            ]
            + ([r["sub"]] if c["factor"] == "background" else [])
        )
        table = "table_texture" in groups
        arms = {v["method"]: v for v in r["arms"]}
        names = ARMS + (WINDOWS if table else ())
        assert list(arms) == list(names)
        n_rollouts += len(names)
        base = arms["frozen"]["trace"]
        first[key] = base[0]
        seeds[key] = [r[v] for v in ["env_seed", "sensor_seed", "policy_seed"]]
        assert seeds[key] == [
            prior[key][v] for v in ["env_seed", "sensor_seed", "policy_seed"]
        ]
        for k, v in arms.items():
            trace = v["trace"]
            n = v["chunks"]
            assert (
                n == v["hook_calls"] == len(trace)
                and [t["step"] for t in trace] == list(range(0, v["steps"], 12))
                and 0 < v["steps"] <= 220
                and type(v["success"]) == bool
            )
            assert all(
                trace[0][n] == base[0][n] for n in ("state", "images", "z")
            ) and all(x["z"] == y["z"] for x, y in zip(trace, base))
            expect = [
                j
                for j in range(n)
                if k != "frozen"
                and (k != "early_0_2" or j < 3)
                and (k != "late_3_plus" or j >= 3)
            ]
            assert v["active_chunks"] == expect
            assert all(z[2] <= 64 and z[3] <= 1 for z in v["roundings"])
            assert len(v["activation"]) == (n if c["factor"] == "clean" else 0)
        if table:
            for k, ref in [("early_0_2", "dead075"), ("late_3_plus", "frozen")]:
                assert arms[k]["trace"][:3] == arms[ref]["trace"][:3]
        for k in ("frozen", "dead075"):
            pv = next(v for v in prior[key]["arms"] if v["method"] == k)
            archive[f"{k}_success_match"] += pv["success"] == arms[k]["success"]
            archive[f"{k}_actions_match"] += pv["action_sha"] == arms[k]["action_sha"]
        wins = {k: int(v["success"]) for k, v in arms.items()}
        rows.append(
            dict(seed=s, id=cid, regime=reg, groups=groups, task=r["task"], wins=wins)
        )
        if table and (s, cid) in seven:
            inj.append(dict(seed=s, id=cid, wins=wins))
        if c["factor"] == "clean":
            for j in (5, 6, 7):

                def get(k):
                    return (
                        arms[k]["activation"][j]
                        if j < len(arms[k]["activation"])
                        else None
                    )

                tele.append(
                    dict(
                        seed=s,
                        id=cid,
                        focus=(s, cid) in focus,
                        chunk=j,
                        frozen=get("frozen"),
                        old_path=get("dead075"),
                        new_path=get("full_traj_dead075"),
                    )
                )
    assert seen == set(prior) and len(inj) == 7 and n_rollouts == 1260
    for (s, cid, reg), u in first.items():
        if reg == "dual":
            v = first[s, cid, "standard"]
            assert (
                u["state"] == v["state"]
                and u["z"] == v["z"]
                and u["images"][1] != v["images"][1]
            )
        if s == 0:
            ks = [(v, cid, reg) for v in range(3)]
            assert (
                len({first[k]["state"] for k in ks}) == 1
                and len({first[k]["z"] for k in ks}) == 3
            )
    grid = {}
    for s in ["all", 0, 1, 2]:
        grid[str(s)] = {}
        for g in GROUPS:
            subset = [
                r for r in rows if g in r["groups"] and (s == "all" or r["seed"] == s)
            ]
            pairs = [(k, "frozen") for k in ARMS] + [("full_traj_dead075", "dead075")]
            if g == "table_texture":
                pairs += [(k, ref) for k in WINDOWS for ref in ("frozen", "dead075")]
            grid[str(s)][g] = {
                k + "_vs_" + ref: stats(subset, k, ref) for k, ref in pairs
            }
    payload = dict(
        status="complete_dev_only",
        source=sources,
        n_rollouts=n_rollouts,
        grid=grid,
        archive_checks=dict(archive),
        prior_seven=inj,
        clean_telemetry=tele,
        rows=rows,
        trials_digest=h.hexdigest(),
    )
    a.out.mkdir(exist_ok=True, parents=True)
    (a.out / "summary.json").write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
    )
    render(payload, a.report)


def render(d, path):
    g = d["grid"]
    lines = [
        "# 第三层全轨迹参考：EGL 配对闭环与桌面时间窗",
        "",
        f"10 条 seed 0 Frozen Clean 提供 {d['source']['n_chunks']} 个实际推理块，外层 9 任务、内层 8+1，按块等权。全局三臂和额外桌面两臂共 1,260 局，另有 10 局源采集。全部为 Dev。",
        "",
    ]
    for s in ["all", "0", "1", "2"]:
        lines += [
            f"## {s}：成功数/分母；救回/误伤；净差 95% 区间（pp）",
            "",
            "|组|同期 Frozen|旧参考 vs Frozen|全轨迹 vs Frozen|全轨迹 vs 旧参考|",
            "|---|---:|---:|---:|---:|",
        ]
        for grp in GROUPS:
            rr = g[s][grp]
            cols = []
            for pair in (
                "frozen_vs_frozen",
                "dead075_vs_frozen",
                "full_traj_dead075_vs_frozen",
                "full_traj_dead075_vs_dead075",
            ):
                v = rr[pair]
                cols.append(
                    f"{v['wins']}/{v['n']}"
                    + (
                        f"；{v['recovered']}/{v['harmed']}；{v['delta_pp']:+.1f} [{v['delta_ci'][0]:.1f}, {v['delta_ci'][1]:.1f}]"
                        if pair != "frozen_vs_frozen"
                        else ""
                    )
                )
            lines.append("|" + grp + "|" + "|".join(cols) + "|")
        lines += [""]
    lines += [
        "## 桌面时间窗：原首帧参考，全部 15 实例 × 3 种子",
        "",
        "|臂|成功|相对 Frozen 救/伤|净数；Δ pp [95% 区间]|",
        "|---|---:|---:|---:|",
    ]
    for k in ("frozen", "dead075", *WINDOWS):
        v = g["all"]["table_texture"][k + "_vs_frozen"]
        lines.append(
            f"|{k}|{v['wins']}/{v['n']}|{v['recovered']}/{v['harmed']}|{v['net']:+d}；{v['delta_pp']:+.1f} {v['delta_ci']}|"
        )
    lines += [
        "",
        "原报告七局误伤（1=成功，0=失败），逐局重跑：",
        "",
        "|ID / 种子|Frozen|旧全程|早期|晚期|新全程|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in d["prior_seven"]:
        lines.append(
            "|"
            + r["id"]
            + " / "
            + str(r["seed"])
            + "|"
            + "|".join(
                str(r["wins"][k])
                for k in ("frozen", "dead075", *WINDOWS, "full_traj_dead075")
            )
            + "|"
        )
    lines += [
        "",
        "## 四局 Clean 腕部通道激活率（%）",
        "",
        "在相同 Frozen 画面比较旧/新参考，同时另报各自策略路径；无此块记 —。不以缺帧当作零。",
        "",
        "|ID / seed|块|同一 Frozen 画面：旧→新|实际旧→新策略路径|",
        "|---|---:|---:|---:|",
    ]

    def act(x, k):
        return "—" if x is None else f"{100 * x[k][1]:.1f}"

    for v in d["clean_telemetry"]:
        if v["focus"]:
            lines.append(
                f"|{v['id']} / {v['seed']}|{v['chunk']}|{act(v['frozen'], 'old')} → {act(v['frozen'], 'full')}|{act(v['old_path'], 'old')} → {act(v['new_path'], 'full')}|"
            )
    lines += [
        "",
        "区间为按该组所覆盖基础任务重采样 30,000 次的描述性配对差区间，同一任务下的所有种子、实例与呈现一起抽；不是新任务或新种子的确认性区间。combined130 重复包含同一批 Sensor 身份。激活率下降不等于恢复抓取；在已有 Dev 上选择过的方法及七局结果不能当作独立盲测。",
        "",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("previous", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--report", type=Path)
    p.add_argument("--sources-only", action="store_true")
    main(p.parse_args())
