#!/usr/bin/env python3
"""Write a true CSV summary from a pure-DINOv2 results.json."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def write_summary_csv(results, path: Path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        path.unlink()
    fields = [
        "feature", "label", "classifier", "control_value",
        "episode_balanced_acc", "episode_macro_f1",
        "frame_balanced_acc", "frame_macro_f1",
        "n_train_episodes", "n_val_episodes",
        "n_train_frames", "n_val_frames",
        "episode_support", "frame_support",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            for key in ("episode_support", "frame_support"):
                row[key] = json.dumps(r.get(key), separators=(",", ":"))
            w.writerow(row)
    return len(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path,
                    default=Path("/hy-tmp/runs/pure_dino_style_native"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root
    data = json.loads((root / "results.json").read_text())
    out = args.out or (root / "summary.csv")
    n = write_summary_csv(data["results"], out)
    print(f"wrote {out} ({n} rows)")
    print(out.read_text().splitlines()[0])
    print(out.read_text().splitlines()[1])

    rows = data["results"]
    with out.open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(rows) == len(csv_rows), (len(rows), len(csv_rows))
    for a, b in zip(rows, csv_rows):
        for key in ("feature", "label", "classifier"):
            assert str(a.get(key)) == b[key], (key, a.get(key), b[key])
        for key in ("episode_balanced_acc", "frame_balanced_acc"):
            assert abs(float(a[key]) - float(b[key])) < 1e-12
    print(f"verified {len(csv_rows)} rows against results.json")


if __name__ == "__main__":
    main()
