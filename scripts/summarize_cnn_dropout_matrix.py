#!/usr/bin/env python3
"""Summarize the official CNN dropout matrix."""
from __future__ import annotations
import argparse, csv, json, math, re
from collections import defaultdict
from pathlib import Path

PAT = re.compile(r"CNN_DROPOUT (\{.*\})")


def _avg_metric(d, prefix):
    pat = re.compile(rf"^view\d+_{prefix}_\d+$")
    vals = [float(v) for k, v in d.items() if pat.match(k)]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--step", type=int, default=50)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    for log in sorted(args.root.glob("*/train.log")):
        run = log.parent.name
        m = re.fullmatch(r"(.+)_s(\d+)", run)
        if not m:
            continue
        group, seed = m.group(1), int(m.group(2))
        found = None
        for line in log.read_text(errors="ignore").splitlines():
            mm = PAT.search(line)
            if not mm:
                continue
            try:
                d = json.loads(mm.group(1))
            except Exception:
                continue
            if int(d.get("step", -1)) == args.step:
                found = d
        mse = None
        train_loss = None
        logs = log.parent / "logs.json.txt"
        if logs.exists():
            for line in logs.read_text(errors="ignore").splitlines():
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("train_action_mse_error") is not None:
                    mse = float(d["train_action_mse_error"])
                if d.get("train_loss") is not None:
                    train_loss = float(d["train_loss"])
        row = {"group": group, "seed": seed, "train_action_mse": mse,
               "train_loss": train_loss}
        if found:
            row["dp_loss"] = found.get("dp_loss")
            row["aux_loss"] = found.get("aux_loss")
            row["keep_frac"] = _avg_metric(found, "keep_frac")
            row["ratio"] = _avg_metric(found, "ratio")
            row["domain_loss"] = _avg_metric(found, "domain_loss")
            row["domain_acc"] = _avg_metric(found, "domain_acc")
            row["feature_norm"] = _avg_metric(found, "feature_norm")
            row["logit_norm"] = _avg_metric(found, "logit_norm")
            row["grl"] = _avg_metric(found, "grl")
        else:
            print(f"[missing step{args.step}] {run}")
        rows.append(row)

    if args.csv is not None:
        fields = ["group", "seed", "train_action_mse", "train_loss",
                  "dp_loss", "aux_loss", "keep_frac", "ratio",
                  "domain_loss", "domain_acc", "feature_norm",
                  "logit_norm", "grl"]
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sorted(rows, key=lambda x: (x["group"], x["seed"])):
                w.writerow({k: r.get(k) for k in fields})
        print(f"wrote {args.csv}")

    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    print(f"runs={len(rows)} step={args.step}")
    for group in sorted(by):
        print(f"\n[{group}] n={len(by[group])}")
        for k in ["train_action_mse", "train_loss", "dp_loss", "aux_loss",
                  "keep_frac", "ratio", "domain_loss", "domain_acc",
                  "feature_norm", "logit_norm", "grl"]:
            vals = [float(x[k]) for x in by[group] if x.get(k) is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            print(f"  {k:18s} mean={mean:+.6f} std={std:.6f} values={vals}")


if __name__ == "__main__":
    raise SystemExit(main())
