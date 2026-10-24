"""独立评测 CoRP-RoboCasa checkpoint：per-dim / per-task MSE（高样本量）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/root/dacsc-robot")
sys.path.insert(0, "/root/dacsc-robot/scripts")

from dacsc.corp_data import LinearNormalizer  # noqa: E402
from dacsc.corp_robocasa import load_adapted_checkpoint  # noqa: E402
from train_corp_robocasa import (  # noqa: E402
    all_task_paths, balanced_loader, build_dataset, evaluate,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-task", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--euler-steps", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--out", type=str, default="/hy-tmp/runs/corp_robocasa_eval/eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, (mean, std) = load_adapted_checkpoint(args.model, device=device)
    norm = LinearNormalizer(mean, std)
    print(f"loaded {args.model}, device={device}", flush=True)

    tasks = all_task_paths(args.max_tasks)
    print(f"tasks={len(tasks)}", flush=True)
    result = {"model": args.model, "per_task": args.per_task, "splits": {}}

    for name, (lo, hi) in (("source", (11, 30)), ("val", (31, 40)), ("test", (41, 50))):
        residues = set(range(9)) if name == "source" else None
        ds = build_dataset(tasks, (lo, hi), args.horizon, 224, True,
                           episode_residues=residues)
        loader = balanced_loader(ds, args.batch_size, args.per_task, seed=123)
        ev, dt = evaluate(model, loader, norm, device, len(loader), args.euler_steps)
        result["splits"][name] = {
            "n_frames_total": len(ds),
            "n_eval_samples": len(loader.dataset),
            "overall": ev["overall"],
            "per_dim": ev["per_dim"],
            "per_task": ev["per_task"],
            "elapsed_sec": dt,
        }
        print(f"{name}: overall={ev['overall']:.4f} n={len(loader.dataset)} ({dt:.1f}s)",
              flush=True)
        print(f"  per_dim: {[ (k, round(v,4)) for k,v in ev['per_dim'].items() ]}",
              flush=True)

    src = result["splits"]["source"]["per_task"]
    for name in ("val", "test"):
        cur = result["splits"][name]["per_task"]
        gap = {t: cur[t] - src[t] for t in src if t in cur}
        result["splits"][name]["per_task_gap_vs_source"] = gap
        ranked = sorted(gap.items(), key=lambda x: x[1], reverse=True)
        print(f"\n{name} worst 10 vs source:")
        for t, g in ranked[:10]:
            print(f"  {t:36s} {g:+.4f}")
        print(f"{name} best 10 vs source:")
        for t, g in ranked[-10:]:
            print(f"  {t:36s} {g:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("DONE ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
