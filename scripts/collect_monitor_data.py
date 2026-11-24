"""Plan/execute the per-step trajectory collection for the monitor.

Source trajectories are for monitor fitting/validation. Target-dev trajectories
are for evaluation only; target test is never collected here.
"""

from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-id", type=int, default=2292)
    ap.add_argument("--roles", default="source_fit,source_validation,target_dev")
    ap.add_argument("--domains", default="clean,shift")
    ap.add_argument("--seeds", default="7777")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.out_root.exists():
        raise FileExistsError(args.out_root)
    args.out_root.mkdir(parents=True)
    plan = []
    for role in args.roles.split(","):
        for domain in args.domains.split(","):
            out = args.out_root / role / domain
            cmd = [sys.executable, str(Path(__file__).with_name("trajectory_collector.py")),
                   "--matrix", str(args.matrix), "--checkpoint", str(args.checkpoint),
                   "--split-json", str(args.split_json), "--out", str(out),
                   "--role", role, "--suite", args.suite,
                   "--lighting-id", str(args.lighting_id), "--seeds", args.seeds,
                   "--domains", domain, "--frame-every", "20", "--device", "cuda:0"]
            plan.append({"role": role, "domain": domain, "out": str(out), "cmd": cmd})
    (args.out_root / "collection_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for item in plan:
        print(json.dumps({"role": item["role"], "domain": item["domain"], "out": item["out"]}))
        if args.execute:
            subprocess.run(item["cmd"], check=True)
    print(json.dumps({"out_root": str(args.out_root), "execute": args.execute, "n": len(plan)}))


if __name__ == "__main__":
    main()
