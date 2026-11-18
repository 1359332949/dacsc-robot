"""Run the monitor training/evaluation pipeline from collected trajectories.

Train on source trajectories only; evaluate on target-dev trajectories only.
Target-test data is never read by this script.
"""

from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path


def run(cmd):
    print(" ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-fit", type=Path, required=True)
    ap.add_argument("--source-val", type=Path, required=True)
    ap.add_argument("--target-dev", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--label-mode", choices=["episode", "within_horizon"], default="episode")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--state-mode", choices=["absolute", "delta"], default="absolute")
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--domain-adversarial", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    scripts = Path(__file__).parent
    source_npz = args.out_dir / "source_windows.npz"
    target_npz = args.out_dir / "target_dev_windows.npz"
    ckpt = args.out_dir / "monitor.pt"
    metrics = args.out_dir / "target_dev_metrics.json"
    commands = [
        [py, str(scripts / "build_monitor_dataset.py"), "--trajectory-dir", str(args.source_fit),
         "--trajectory-dir", str(args.source_val), "--out", str(source_npz),
         "--window", str(args.window), "--stride", str(args.stride),
         "--label-mode", args.label_mode, "--horizon", str(args.horizon),
         "--state-mode", args.state_mode],
        [py, str(scripts / "train_failure_monitor.py"), "--data", str(source_npz),
         "--out", str(ckpt), "--epochs", str(args.epochs), "--window", str(args.window)]
        + (["--no-visual"] if args.no_visual else [])
        + (["--domain-adversarial"] if args.domain_adversarial else []),
        [py, str(scripts / "evaluate_monitor_offline.py"), "--data", str(target_npz),
         "--checkpoint", str(ckpt), "--out", str(metrics),
         "--source", str(source_npz)],
    ]
    # Build target windows without training on them.
    commands.insert(1, [py, str(scripts / "build_monitor_dataset.py"),
                        "--trajectory-dir", str(args.target_dev), "--out", str(target_npz),
                        "--window", str(args.window), "--stride", str(args.stride),
                        "--label-mode", args.label_mode, "--horizon", str(args.horizon),
                        "--state-mode", args.state_mode])
    for cmd in commands:
        if args.execute:
            run(cmd)
        else:
            print(" ".join(str(x) for x in cmd))
    print(json.dumps({"ok": True, "execute": args.execute, "source_windows": str(source_npz),
                      "target_windows": str(target_npz), "checkpoint": str(ckpt),
                      "metrics": str(metrics)}))


if __name__ == "__main__":
    main()
