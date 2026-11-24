"""Quantify paired trajectories on the archive-identical prefix only."""

import argparse, hashlib, json
from pathlib import Path
from collections import Counter
import numpy as np
from analyze_block3_table_harm import select, ARCHIVE


def first(a, t):
    return next((i for i, v in enumerate(a) if v >= t), None)


def phase(i):
    return "未达" if i is None else "早" if i < 3 else "中" if i < 8 else "晚"


def fmt(x):
    return "—" if x is None else str(x)


def vec(x):
    return "/".join(f"{k:.2f}" for k in x)


def view(x):
    return f"{vec(np.array(x['active']) * 100)}%；{vec(x['scale'])}"


def run(p):
    selected = select(p.archive)
    cases = []
    lines = []
    for group, r in selected:
        f = p.replay / f"{r['case_id']}_{r['seed']}.json"
        blob = json.loads(f.read_text())
        assert (blob["group"], blob["id"], blob["seed"], blob["task"]) == (
            group,
            r["case_id"],
            r["seed"],
            r["task"],
        )
        arms = blob["arms"]

        def prefix(v):
            return next(
                (i for i, c in enumerate(v["trace"]) if not c["verified"]),
                len(v["trace"]),
            )

        # Only the hash match, never an outcome or metric, ranks retry attempts.
        attempts = [("replay", arms["frozen"])]
        for fp in sorted(p.replay.parent.glob("retry*/" + f.name)):
            other = json.loads(fp.read_text())
            assert other["id"] == blob["id"] and other["seed"] == blob["seed"]
            attempts.append((fp.parent.name, other["arms"]["frozen"]))
        source, arms["frozen"] = max(
            attempts, key=lambda pair: (pair[1]["full_match"], prefix(pair[1]))
        )
        frozen = arms["frozen"]
        adapted = arms["dead075"]
        fs, ad = frozen["trace"], adapted["trace"]
        shared = []
        lossless = []
        for arm in ("frozen", "dead075"):
            ar = next(x for x in r["arms"] if x["method"] == arm)
            v = arms[arm]
            lossless.append(v["full_match"])
            ah = hashlib.sha256()
            for j, t in enumerate(v["trace"]):
                plan = np.ascontiguousarray(t["plan"], dtype=np.float32)
                assert plan.shape == (12, 7)
                if t["verified"]:
                    assert (
                        hashlib.sha256(
                            str(plan.dtype).encode()
                            + str(plan.shape).encode()
                            + plan.tobytes()
                        ).hexdigest()
                        == ar["trace"][j]["plan"]
                    )
                ah.update(plan[: min(12, max(0, v["steps"] - 12 * j))].tobytes())
            if v["full_match"]:
                assert ah.hexdigest() == ar["action_sha"]
            if v["full_match"]:
                assert len(ar["trace"]) == len(v["trace"]) and (
                    ar["steps"],
                    ar["success"],
                ) == (v["steps"], v["success"])
        for i, (x, y) in enumerate(zip(fs, ad)):
            if not x["verified"] or not y["verified"]:
                break
            a, b = (np.array(v["plan"]) for v in (x, y))
            assert a.shape == b.shape == (12, 7)
            d = a - b
            norm = lambda d: float(np.linalg.norm(d) / np.sqrt(12))
            w = lambda v: {k: v[k] for k in ("active", "scale")}
            shared.append(
                dict(
                    chunk=i,
                    action_l2=float(np.linalg.norm(d)),
                    action=norm(d),
                    xyz=norm(d[:, :3]),
                    rot=norm(d[:, 3:6]),
                    grip=norm(d[:, 6]),
                    eef_cm=float(100 * np.linalg.norm(np.subtract(x["eef"], y["eef"]))),
                    bowl_cm=[
                        float(100 * np.linalg.norm(np.subtract(v["eef"], v["bowl"])))
                        for v in (x, y)
                    ],
                    frozen=w(x),
                    adapted=w(y),
                )
            )
        exact = next(
            (
                i
                for i, (u, v) in enumerate(
                    zip(r["arms"][0]["trace"], r["arms"][2]["trace"])
                )
                if u["plan"] != v["plan"]
            ),
            None,
        )
        assert shared, "no verified common chunks"
        A = [v["action"] for v in shared]
        E = [v["eef_cm"] for v in shared]
        ai = first(A, 0.25)
        ei = first(E, 1.0)
        ji = next((i for i in range(len(A)) if A[i] >= 0.25 and E[i] >= 1), None)
        xi = first([c["xyz"] for c in shared], 0.10)
        near = int(np.argmin([v["bowl_cm"][1] for v in shared]))
        assert adapted["full_match"]
        # For one-arm observables no Frozen float telemetry is required. Limit the
        # window using the archived arm lengths, not the replay prefix.
        n_common = min(len(r["arms"][0]["trace"]), len(r["arms"][2]["trace"]))
        aseries = [
            dict(
                active=v["active"],
                scale=v["scale"],
                bowl_cm=float(100 * np.linalg.norm(np.subtract(v["eef"], v["bowl"]))),
            )
            for v in ad[:n_common]
        ]
        anear = int(np.argmin([v["bowl_cm"] for v in aseries]))
        cases.append(
            dict(
                group=group,
                id=r["case_id"],
                seed=r["seed"],
                task=r["task"],
                exact_first=exact,
                action_first=ai,
                eef_first=ei,
                joint_first=ji,
                xyz_first=xi,
                near=near,
                action_sweep=[first(A, t) for t in (0.125, 0.25, 0.50)],
                eef_sweep=[first(E, t) for t in (0.5, 1.0, 2.0)],
                shared=shared,
                adapted_common=aseries,
                adapted_near=anear,
                full_match=lossless,
                frozen_source=source,
                attempts={k: prefix(v) for k, v in attempts},
                arm_chunks=[len(fs), len(ad)],
            )
        )
    assert len(cases) == 13
    p.out.mkdir(parents=True, exist_ok=True)
    (p.out / "summary.json").write_text(
        json.dumps(
            dict(archive=ARCHIVE, cases=cases), separators=(",", ":"), allow_nan=False
        )
        + "\n"
    )
    lines = [
        "# Block 3：桌面误伤的逐块归因检查",
        "",
        "Frozen 和 α=0.75；只回放预定的 13 对。原包仅含哈希，浮点数来自校验匹配的定点回放，不补齐较早结束后的数据。第 0 块在机器人执行首个动作前。时间段只表示块号，不是实际抓取事件标注。",
        "",
        "**判决：正常近景被首帧参考标成异常，确实发生了；但数据不支持它是桌面误伤的共同起点。** 7 局误伤（5 个实例，3 个基础任务）中，6 局在块 0–2 已有 ≥1 cm 的 EEF 偏差，另一局在块 4 达到。把距离线换成 0.5 / 2 cm，早期达到的分别为 7 / 2 局，‘较早已有小偏差’比‘多数已发生大偏离’更稳妥。两局救回也是块 1 达到 1 cm，且是同一实例的两个种子。因而早偏不等于会失败。",
        "",
        "桌面误伤在首帧就激活主/腕通道中位 73.7% / 75.0%；到首次 1 cm EEF 偏差，腕部激活中位变化 +3.12 pp；到最近碗心的共同时间点为 +0.52 pp，7 局没有一局增加 10 pp。部分局在其他时刻涨得多，且高起点限制了上升空间。两局救回的首帧腕部激活甚至是 99.74%。激活不是失败检测器。Clean 误伤在块 6 首次偏开 1 cm，腕部激活从 11.20% 到 93.23%；3 局两臂均成功的 Clean 在相同判据下也全部上升至少 10 pp（中位 +44.79 pp）。",
        "",
        "13 条干预臂、12 条 Frozen 与原包全程吻合。剩余 `95599a32…/seed1` 的 Frozen 在块 3 开始图像、计划哈希不同，四次定点重试均未重现；仅使用块 0–2。该局首次 1 cm（块 2）可确定，首次全动作及联合过线时间右删失，即后续未知。独立于 Frozen 数值的干预臂统计可使用全部原始共同时间窗。",
        "",
        "**下一步：先检验早期偏差是否参与造成误伤。** 在既定 Dev 桌面实例（包括成败不变者）上固定 α、LOTO、种子，只改变对齐施加的时间：仅块 0–2 与从块 3 起，和已有 Frozen/全程对齐配对。它是机制消融，不是给桌面写上线规则。全轨迹 Clean 参考值得另做单变量对照，但这批观察不能保证换参考就会修复；也不能证明局部纹理通过某些通道导致失败。",
        "",
        "## 每一对何时偏开",
        "",
        "A = 12 步 × 7 分量计划动作 L2 / √12 首次 ≥0.25（控制量，包括夹爪）；E = 块首 EEF 距离首次 ≥1 cm；联合要求同一块同时达到两者。XYZ 是只看平移命令的 RMS ≥0.10。不是统计显著性检验。",
        "",
        "|组 / ID / 种子|共同可验证块数|首个不同动作哈希|A 块|XYZ 块|E 块|联合块 / 时段|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        lines.append(
            f"|{c['group']} / `{c['id']}` / {c['seed']}|{len(c['shared'])}|{fmt(c['exact_first'])}|{fmt(c['action_first'])}|{fmt(c['xyz_first'])}|{fmt(c['eef_first'])}|{fmt(c['joint_first'])} / {('后续未知' if c['joint_first'] is None and not all(c['full_match']) else phase(c['joint_first']))}|"
        )
    lines += [
        "",
        "## 第 3 层的两个视角",
        "",
        "每格为主/腕激活通道百分比；主/腕平均 |γ−1|（无量纲，保留三位，含未激活通道的零值）。以下为实际干预臂，Frozen 路径同位置的假想算子统计在 JSON；这组反事实测量没有改变 Frozen 动作。",
        "",
        "|组 / ID 前缀 / 种子|第 0 块|首个 A 块|首个 E 块|离碗最近块及该块统计|",
        "|---|---|---|---|---|",
    ]

    def vw(v):
        return (
            vec(np.array(v["active"]) * 100)
            + "%；"
            + "/".join(f"{z:.3f}" for z in v["scale"])
        )

    for c in cases:
        d = c["shared"]
        val = lambda i: "—" if i is None else vw(d[i]["adapted"])
        lines.append(
            f"|{c['group']} / `{c['id'].removeprefix('clean_')[:8]}` / {c['seed']}|{val(0)}|{val(c['action_first'])}|{val(c['eef_first'])}|{c['adapted_near']}：{vw(c['adapted_common'][c['adapted_near']])}|"
        )
    lines += [
        "",
        "## 激活上升是否特属于误伤",
        "",
        "以下为相对本局第 0 块的**腕部激活比例差（百分点）**，及平均有效缩放幅度差。对每组先按轨迹作差，再取中位数。最近指该干预臂在共同观察窗口内 EEF 离碗中心最近，不等同于相机距离或已抓住。",
        "",
        "|组别|对照时刻|可用局数|腕部激活变化中位数 (pp)|腕部缩放变化中位数|上升 ≥10 pp 局数|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for g in ["table_harm", "table_rescue", "clean_harm", "clean_control"]:
        group = [c for c in cases if c["group"] == g]
        print(
            g,
            "action/eef/joint",
            [
                [c[k] for c in group]
                for k in ("action_first", "eef_first", "joint_first")
            ],
        )
        for label in ("action_first", "eef_first", 3, 5, 7, 8, "near"):
            delta = []
            scale = []
            for c in group:
                i = label if isinstance(label, int) else c[label]
                d = c["adapted_common"]
                if label == "near":
                    i = c["adapted_near"]
                if i is None or i >= len(d):
                    continue
                a = d[i]
                b = d[0]
                delta.append(100 * (a["active"][1] - b["active"][1]))
                scale.append(a["scale"][1] - b["scale"][1])
            if delta:
                lines.append(
                    f"|{g}|{label}|{len(delta)}|{np.median(delta):+.2f}|{np.median(scale):+.4f}|{sum(v >= 10 for v in delta)}|"
                )
    lines += [
        "",
        "完整 JSON 给出每个共同块的动作 L2、XYZ/旋转/夹爪分量、EEF 间距、两臂到碗心距离、两路激活和缩放，以及阈值敏感性。选样依赖结局，且病例可跨种子重复，因此不以病例数作独立因果显著性证据。",
        "",
    ]
    p.report.write_text("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("replay", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("report", type=Path)
    run(p.parse_args())
