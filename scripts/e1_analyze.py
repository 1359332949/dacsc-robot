"""Summarize E1 frozen degradation screen (screening only, 10 episodes)."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    recs = [json.loads(x) for x in (args.dir / "e1_frozen.jsonl").read_text().splitlines() if x.strip()]
    clean = {}
    for r in recs:
        if r["domain"] == "clean":
            clean.setdefault((r["suite"], r["base_task"]), {})[r["init_index"]] = r
    cells = {}
    for r in recs:
        if r["domain"] == "clean": continue
        key = (r["suite"], r["base_task"], r["domain"], r["variant_id"])
        cells.setdefault(key, []).append(r)
    out = {"screening": True, "episodes_per_cell": None, "clean": {}, "shift": {}}
    for k, rows in clean.items():
        out["clean"][f"{k[0]}::{k[1]}"] = {
            "n": len(rows), "successes": sum(int(r["success"]) for r in rows.values()),
            "success_rate": sum(int(r["success"]) for r in rows.values()) / len(rows)}
    for key, rows in sorted(cells.items()):
        suite, base, domain, vid = key
        base_clean = clean[(suite, base)]
        n = len(rows)
        pair = []
        for r in rows:
            c = base_clean.get(r["init_index"])
            if c is not None:
                pair.append(int(r["success"]) - int(c["success"]))
        sr = sum(int(r["success"]) for r in rows) / n
        csr = (sum(int(base_clean[r["init_index"]]["success"]) for r in rows if r["init_index"] in base_clean) /
               max(1, sum(1 for r in rows if r["init_index"] in base_clean)))
        boot = []
        if pair:
            rs = np.random.RandomState(1234)
            for _ in range(5000):
                boot.append(float(np.mean(rs.choice(pair, size=len(pair), replace=True))))
        out["shift"][f"{suite}::{base}::{domain}::{vid}"] = {
            "n": n, "successes": sum(int(r["success"]) for r in rows),
            "success_rate": sr, "paired_clean_success_rate": csr,
            "delta_sr": sr - csr, "paired_delta_mean": float(np.mean(pair)) if pair else None,
            "paired_delta_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None,
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
        }
    out["episodes_per_cell"] = max([v["n"] for v in out["shift"].values()] or [0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"clean": out["clean"], "shift": out["shift"]}, indent=2))


if __name__ == "__main__":
    main()
