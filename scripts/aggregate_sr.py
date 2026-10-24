"""Aggregate per-task success JSONs into split-level success rates."""
from __future__ import annotations
import json
import sys
from pathlib import Path


def agg(dir_path: Path) -> dict:
    n = s = 0
    tasks = {}
    for f in sorted(dir_path.glob("*.json")):
        d = json.loads(f.read_text())
        n += int(d.get("episodes", 0))
        s += int(d.get("successes", 0))
        tasks[d.get("task", f.stem)] = {
            "episodes": d.get("episodes", 0),
            "successes": d.get("successes", 0),
            "success_rate": d.get("success_rate", 0.0),
        }
    return {"episodes": n, "successes": s,
            "success_rate": (s / n) if n else 0.0, "tasks": tasks}


def main() -> int:
    root = Path(sys.argv[1])
    out = {}
    for split in ("source", "val", "test"):
        d = root / split
        if d.is_dir():
            out[split] = agg(d)
    if "source" in out and "test" in out:
        out["gap_source_minus_test"] = out["source"]["success_rate"] - out["test"]["success_rate"]
    (root / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (v if k == "gap_source_minus_test" else
                          {kk: vv for kk, vv in v.items() if kk != "tasks"})
                      for k, v in out.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
