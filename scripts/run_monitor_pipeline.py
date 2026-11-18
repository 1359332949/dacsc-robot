"""Train/evaluate a monitor with explicit source-fit/val/target-dev roles."""

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
    ap.add_argument("--state-mode", choices=["absolute", "delta"], default="delta")
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--domain-adversarial", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    scripts = Path(__file__).parent
    fit_npz = args.out_dir / "source_fit_windows.npz"
    val_npz = args.out_dir / "source_val_windows.npz"
    target_npz = args.out_dir / "target_dev_windows.npz"
    ckpt = args.out_dir / "monitor.pt"
    metrics = args.out_dir / "target_dev_metrics.json"

    def build(src, out):
        return [py, str(scripts / "build_monitor_dataset.py"), "--trajectory-dir", str(src),
                "--out", str(out), "--window", str(args.window), "--stride", str(args.stride),
                "--label-mode", args.label_mode, "--horizon", str(args.horizon),
                "--state-mode", args.state_mode]

    commands = [
        build(args.source_fit, fit_npz),
        build(args.source_val, val_npz),
        build(args.target_dev, target_npz),
        [py, str(scripts / "train_failure_monitor_roles.py"),
         "--train-data", str(fit_npz), "--val-data", str(val_npz),
         "--out", str(ckpt), "--epochs", str(args.epochs), "--window", str(args.window)]
        + (["--no-visual"] if args.no_visual else [])
        + (["--domain-adversarial"] if args.domain_adversarial else []),
        [py, str(scripts / "evaluate_monitor_offline.py"), "--data", str(target_npz),
         "--checkpoint", str(ckpt), "--out", str(metrics),
         "--source", str(val_npz), "--source-all"],
    ]
    for cmd in commands:
        if args.execute: run(cmd)
        else: print(" ".join(str(x) for x in cmd))
    print(json.dumps({"ok": True, "execute": args.execute, "source_fit_windows": str(fit_npz),
                      "source_val_windows": str(val_npz), "target_dev_windows": str(target_npz),
                      "checkpoint": str(ckpt), "metrics": str(metrics)}))


if __name__ == "__main__":
    main()
