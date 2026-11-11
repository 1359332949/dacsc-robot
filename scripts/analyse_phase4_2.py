"""Analyse multi-domain experiments: paired success, McNemar, cluster bootstrap."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/hy-tmp/validation/p42")
OUT = Path("/hy-tmp/validation/p42_analysis")
METHODS = ["frozen", "salt", "sg_cpa", "sg_noproj", "sg_excl"]
DOMAINS = ["Camera", "Background", "Noise", "Light"]


def load(method):
    rows = []
    for f in sorted((ROOT / method).glob("*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def task_domain_map():
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/phase3_final_tasks.json").read_text())
    m = {}
    for cell, info in cfg["cells"].items():
        for t in info["tasks"]:
            m[(info["suite"], int(t))] = info["domain"]
    return m


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def bootstrap_task_diff(tasks, keys_by_task, a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sample = [tasks[i] for i in rng.integers(0, len(tasks), len(tasks))]
        d = 0
        n = 0
        for t in sample:
            for k in keys_by_task[t]:
                d += int(b[k]["success"]) - int(a[k]["success"])
                n += 1
        diffs.append(d / max(n, 1))
    return [float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmap = task_domain_map()
    data = {m: load(m) for m in METHODS if (ROOT / m).exists()}
    idx = {m: {(r["suite"], r["task_id"], r.get("episode_id", 0)): r for r in rows}
           for m, rows in data.items()}
    common = set.intersection(*[set(v) for v in idx.values()]) if idx else set()
    print("episodes per method:", {m: len(idx[m]) for m in idx}, "common:", len(common))
    rows = []
    for dom in DOMAINS + ["ALL"]:
        keys = sorted(k for k in common if dom == "ALL" or tmap.get((k[0], k[1])) == dom)
        if not keys:
            continue
        rec = {"domain": dom, "n": len(keys)}
        for m in idx:
            rec[f"sr_{m}"] = sum(int(idx[m][k]["success"]) for k in keys) / len(keys)
        for a, b in [("salt", "sg_cpa"), ("frozen", "sg_cpa"), ("frozen", "salt"),
                     ("sg_noproj", "sg_cpa"), ("sg_excl", "sg_cpa"),
                     ("frozen", "sg_excl"), ("salt", "sg_excl"),
                     ("frozen", "sg_noproj")]:
            if a not in idx or b not in idx:
                continue
            b01 = sum(1 for k in keys if not idx[a][k]["success"] and idx[b][k]["success"])
            b10 = sum(1 for k in keys if idx[a][k]["success"] and not idx[b][k]["success"])
            rec[f"mcnemar_{b}_vs_{a}"] = {"b01": b01, "b10": b10,
                                          "p": mcnemar(b01, b10),
                                          "pair_diff": (b01 - b10) / len(keys)}
        tasks = sorted({(k[0], k[1]) for k in keys})
        keys_by_task = {t: [k for k in keys if (k[0], k[1]) == t] for t in tasks}
        if len(tasks) >= 2:
            for a, b in (("salt", "sg_cpa"), ("frozen", "sg_cpa"),
                         ("salt", "sg_excl"), ("frozen", "sg_excl"),
                         ("frozen", "salt")):
                if a in idx and b in idx:
                    rec[f"{b}_minus_{a}_ci90"] = bootstrap_task_diff(
                        tasks, keys_by_task, idx[a], idx[b])
        rows.append(rec)
    # mechanism metrics
    mech = {}
    for m in ("sg_cpa", "sg_noproj", "sg_excl"):
        if m not in data:
            continue
        eps = [r for r in data[m] if r.get("sg")]
        if not eps:
            continue
        mech[m] = {
            "n": len(eps),
            "success": sum(int(r["success"]) for r in eps) / len(eps),
            "updates_mean": float(np.mean([r["updates"] for r in eps])),
            "retain_mean": float(np.mean([r["sg"]["retain_mean"] for r in eps])),
            "accepted_rate": float(np.mean([r["sg"]["accepted_rate"] for r in eps])),
            "backtrack_rate": float(np.mean([r["sg"]["backtrack_rate"] for r in eps])),
            "beta_abs_max_mean": float(np.mean([r["sg"]["beta_abs_max_last"] for r in eps])),
            "replan_seconds_mean": float(np.mean([r["sg"]["replan_seconds_mean"] for r in eps])),
        }
    res = {"rows": rows, "mechanism": mech,
           "episode_counts": {m: len(v) for m, v in idx.items()}}
    (OUT / "phase4_2_analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(rows, indent=2))
    print(json.dumps(mech, indent=2))


if __name__ == "__main__":
    main()
