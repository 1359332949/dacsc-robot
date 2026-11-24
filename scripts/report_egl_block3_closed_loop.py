"""Audit the entire paired grid before publishing a score, then bootstrap whole tasks."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/data"
ARMS = ("frozen", "dead050", "dead075")
GROUPS = (
    "clean",
    "dual_sensor",
    "single_sensor",
    "background",
    "table_texture",
    "scene_texture",
    "lighting",
    "standard100",
    "combined130",
)


def score(rows):
    tasks = sorted({r["task"] for r in rows})
    counts = np.array(
        [
            [
                sum(r["task"] == t for r in rows),
                *[sum(r["wins"][i] for r in rows if r["task"] == t) for i in range(3)],
            ]
            for t in tasks
        ]
    )
    n = len(rows)
    assert counts[:, 0].sum() == n
    signed = np.array([r["wins"] for r in rows], int)
    w = np.random.default_rng(20261124).multinomial(
        len(tasks), np.full(len(tasks), 1 / len(tasks)), size=30000
    )
    boot = w @ counts
    rates = 100 * boot[:, 1:] / boot[:, :1]
    ans = {}
    for i, k in enumerate(ARMS):
        d = signed[:, i] - signed[:, 0]
        ci = lambda x: np.quantile(x, [0.025, 0.975]).round(3).tolist()
        wins = int(signed[:, i].sum())
        ans[k] = dict(
            n=n,
            tasks=len(tasks),
            wins=wins,
            sr_pct=round(100 * wins / n, 3),
            sr_ci=ci(rates[:, i]),
            recovered=int((d > 0).sum()),
            harmed=int((d < 0).sum()),
            net=int(d.sum()),
            delta_pp=round(100 * d.mean(), 3),
            delta_ci=ci(rates[:, i] - rates[:, 0]),
        )
    return ans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    mp = DOC / "static_visual_v2_suite/final/dev_manifest.json"
    cases = {c["case_id"]: c for c in json.loads(mp.read_text())["cases"]}
    expect = {
        (s, c["case_id"], reg)
        for s in (0, 1, 2)
        for c in cases.values()
        for reg in (
            ["standard", "dual"] if c["factor"] == "sensor_noise" else ["standard"]
        )
    }
    assert len(expect) == 390
    archive = {
        (s, r["case_id"]): r
        for s in (0, 1, 2)
        for r in json.loads((DOC / f"egl_baseline_seed{s}.json").read_text())["rows"]
    }
    exposed = set()
    compact = []
    old = Counter()
    mismatch = []
    roundings = Counter()
    hashes = hashlib.sha256()
    first = {}
    streams = {}
    for path in sorted(a.root.glob("*.json")):
        data = path.read_bytes()
        hashes.update(path.name.encode() + hashlib.sha256(data).digest())
        r = json.loads(data)
        s = r["seed"]
        cid = r["case_id"]
        reg = r["regime"]
        key = (s, cid, reg)
        assert key in expect - exposed
        exposed.add(key)
        streams[key] = (r["env_seed"], r["sensor_seed"], r["policy_seed"])
        assert (
            r["reference"]
            == "9dceecd7ac5d343b49f2c2c7808becbd50ec6ed90b8077b7bd54c57e66f0f44f"
        )
        c = cases[cid]
        task = c["base_task"]
        sub = c["sub_dimension"]
        f = c["factor"]
        assert r["task"] == task and r["factor"] == f and r["sub"] == sub
        rr = r["arms"]
        assert [x["method"] for x in rr] == list(ARMS)
        base = rr[0]["trace"]
        wins = []
        for i, x in enumerate(rr):
            trace = x["trace"]
            assert (
                len(trace) == x["chunks"]
                and x["hook_calls"] == (len(trace) if i else 0)
                and [t["step"] for t in trace] == list(range(0, x["steps"], 12))
                and 0 < x["steps"] <= 220
            )
            assert type(x["success"]) == bool
            assert all(
                trace[0][k] == base[0][k] for k in ("state", "images", "z")
            ) and all(t["z"] == u["z"] for t, u in zip(trace, base))
            assert all(v[2] <= 64 and v[3] <= 1 for v in x["roundings"])
            roundings[x["method"]] += len(x["roundings"])
            wins.append(int(x["success"]))
        if reg == "standard":
            leg = archive[s, cid]
            old["matching_success"] += wins[0] == leg["success"]
            old["matching_steps"] += rr[0]["steps"] == leg["steps"]
            old["matching_agent_pixels"] += (
                base[0]["images"][0] == leg["init_agentview_sha256"]
            )
            if wins[0] != leg["success"]:
                mismatch.append([s, cid, int(leg["success"]), wins[0]])
        first[key] = base[0]
        groups = ["combined130"] + (
            ["dual_sensor"]
            if reg == "dual"
            else ["standard100", "single_sensor" if f == "sensor_noise" else f]
            + ([sub] if f == "background" else [])
        )
        compact.append(dict(seed=s, id=cid, task=task, groups=groups, wins=wins))
    assert exposed == expect, (len(exposed), len(expect - exposed))
    checks = Counter()
    presentations = {(i, reg) for _, i, reg in expect}
    for cid, reg in presentations:
        kk = [(s, cid, reg) for s in (0, 1, 2)]
        assert len({(streams[k][:2], first[k]["state"]) for k in kk}) == 1
        assert len({streams[k][2] for k in kk}) == len({first[k]["z"] for k in kk}) == 3
    for (s, cid, reg), u in first.items():
        if reg == "dual":
            v = first[s, cid, "standard"]
            assert u["state"] == v["state"] and u["z"] == v["z"]
            checks["identical_agent_hash"] += u["images"][0] == v["images"][0]
            checks["changed_wrist_hash"] += u["images"][1] != v["images"][1]
    res = {}
    by_task = {}
    for seed in ("all", 0, 1, 2):
        res[str(seed)] = {}
        for group in GROUPS:
            rows = [
                r
                for r in compact
                if group in r["groups"] and (seed == "all" or r["seed"] == seed)
            ]
            res[str(seed)][group] = score(rows)
            if seed == "all":
                by_task[group] = {
                    t: np.array([r["wins"] for r in rows if r["task"] == t])
                    .sum(0)
                    .tolist()
                    for t in sorted({r["task"] for r in rows})
                }
    decision = {}
    for method in ARMS[1:]:
        pooled = res["all"]
        clean = pooled["clean"][method]
        domains = ("dual_sensor", "background", "lighting")
        decision[method] = dict(
            clean_net=clean["net"],
            clean_harmed=clean["harmed"],
            positive_domains=[g for g in domains if pooled[g][method]["net"] > 0],
            all_three_positive=all(pooled[g][method]["net"] > 0 for g in domains),
            positive_every_seed=[
                g
                for g in domains
                if all(res[str(s)][g][method]["net"] > 0 for s in range(3))
            ],
            delta_ci_above_zero=[
                g for g in domains if pooled[g][method]["delta_ci"][0] > 0
            ],
        )
    result = dict(
        decision=decision,
        grid=res,
        task_success_counts=by_task,
        paired_outcomes=compact,
        arm_order=ARMS,
        raw_digest=hashes.hexdigest(),
        verified_three_seed_presentations=len(presentations),
        archive_standard_frozen_matches=dict(old),
        archive_success_disagreements=mismatch,
        rounding_events=dict(roundings),
        dual_vs_single_first_frame=dict(checks),
    )
    a.out.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    lines = [
        "# 第 3 层死区对齐：EGL 三种子闭环 Dev 验证",
        "",
        "每行使用同一组真实重跑的 Frozen 和两个固定强度。共 390 个配对条件、1,170 局；额外 pilot 不计分。表格的百分比为闭环成功率，差值区间为相对同期 Frozen 的百分点。每格格式：成功数/总数（成功率）；救回/误伤，净数，[配对差 95% 区间]。",
        "",
    ]
    for method in ARMS[1:]:
        d = decision[method]
        lines += [
            f"**{method}**：Clean 净数 {d['clean_net']:+d}（误伤 {d['clean_harmed']} 局）；双眼 Sensor、背景和灯光中，合计净正的域：{', '.join(d['positive_domains']) or '无'}；三个种子分别均为净正的域：{', '.join(d['positive_every_seed']) or '无'}；配对差区间下界超过零的域：{', '.join(d['delta_ci_above_zero']) or '无'}。",
            "",
        ]
    for ss, grid in res.items():
        lines += [
            f"## {'三种子合计' if ss == 'all' else '策略种子 ' + ss}",
            "",
            "|组别|Frozen|α=0.50（主候选）|α=0.75（对照）|",
            "|---|---:|---:|---:|",
        ]
        for g, v in grid.items():
            cells = []
            for i, h in enumerate(ARMS):
                z = v[h]
                text = f"{z['wins']}/{z['n']} ({z['sr_pct']:.1f}%)"
                cells.append(
                    text
                    + (
                        f"；{z['recovered']}/{z['harmed']}，{z['net']:+d}，[{z['delta_ci'][0]:.1f}, {z['delta_ci'][1]:.1f}]"
                        if i
                        else ""
                    )
                )
            lines.append("|" + g + "|" + "|".join(cells) + "|")
        lines += [""]
    lines += [
        "标准 100 与双眼 30 不是两个独立数据集，后者复用标准 Sensor 的身份。combined130 是描述性总计。每域按实际覆盖的基础任务有放回重采样 30,000 次；同一任务的所有实例、相机条件和种子一起抽。10 个任务以内的区间仅用于描述。JSON 同时保存各臂成功率区间、各任务计数、每一对胜败和标准 Frozen 历史一致性检查。",
        "",
        f"300 条标准 Frozen 与历史成功标签一致 {old['matching_success']}/300；步数一致 {old['matching_steps']}/300。所有 390 个三臂组首帧双眼、模拟器状态与首个策略噪声一致，共同步数内 z 流一致。不同强度在每个实际推理 chunk 均执行一次干预。",
        "",
    ]
    a.report.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
