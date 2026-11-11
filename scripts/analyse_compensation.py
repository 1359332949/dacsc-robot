"""Analyse the compensation probe: is the action-subspace shift larger on successes?"""

import json

import numpy as np
from scipy.stats import mannwhitneyu


def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return rows


def summary(rows, label):
    succ = [r["a_shift_mean"] for r in rows if r["success"]]
    fail = [r["a_shift_mean"] for r in rows if not r["success"]]
    rs = [r["r_shift_mean"] for r in rows if r["success"]]
    rf = [r["r_shift_mean"] for r in rows if not r["success"]]
    print(f"{label}: n={len(rows)} success={len(succ)} fail={len(fail)}")
    if succ:
        print(f"  A-shift  success mean={np.mean(succ):.3f} median={np.median(succ):.3f}")
    if fail:
        print(f"  A-shift  fail    mean={np.mean(fail):.3f} median={np.median(fail):.3f}")
    if succ and fail:
        u = mannwhitneyu(succ, fail, alternative="greater")
        print(f"  Mann-Whitney (success > fail) p={u.pvalue:.4f}")
    if rs:
        print(f"  R-shift  success mean={np.mean(rs):.3f}")
    if rf:
        print(f"  R-shift  fail    mean={np.mean(rf):.3f}")


def main():
    cam = load("/hy-tmp/validation/comp_probe/cam.jsonl")
    clean = load("/hy-tmp/validation/comp_probe_clean/clean.jsonl")
    summary(cam, "camera-corrupted")
    summary(clean, "clean control")
    # within-task: tasks with both success and failure episodes
    by_task = {}
    for r in cam:
        by_task.setdefault(r["task_id"], []).append(r)
    diffs = []
    for tid, rows in by_task.items():
        s = [x["a_shift_mean"] for x in rows if x["success"]]
        f = [x["a_shift_mean"] for x in rows if not x["success"]]
        if s and f:
            diffs.append(np.mean(s) - np.mean(f))
    if diffs:
        print(f"within-task (success-fail) A-shift diff: mean={np.mean(diffs):.3f} n_tasks={len(diffs)}")
    else:
        print("within-task: no tasks with both outcomes")


if __name__ == "__main__":
    main()
