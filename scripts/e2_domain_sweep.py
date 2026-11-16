"""Run E2 capture + Resampler Oracle diagnostics for every matrix variant.

No training; frozen CoRP.  Parallelizes independent conditions with a process
pool.  One condition = v12 capture then v2 repair diagnostic.
"""

from __future__ import annotations
import argparse, datetime, hashlib, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).is_file() else None


def clean_language(task_name: str) -> str:
    """Official LIBERO clean-task instruction.

    LIBERO-10 names carry a scene prefix such as KITCHEN_SCENE3_; the
    benchmark language drops that prefix.  Other suites use the underscore
    name verbatim.
    """
    parts = task_name.split("_")
    if parts and parts[0].startswith("KITCHEN") and len(parts) >= 3 and parts[1].startswith("SCENE"):
        return " ".join(parts[2:])
    return " ".join(parts)


def tag_for(task, domain, cid):
    safe = task["base_task"].replace("/", "_")
    return f"{task['suite']}__{safe}__{domain}__{cid}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--suites", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--num-states", type=int, default=50)
    ap.add_argument("--checkpoint", type=Path, default=Path("/hy-tmp/dacsc-robot/repos/CoRP/checkpoints/corp_libero.pt"))
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    matrix = json.loads(args.matrix.read_text())
    conds = []
    for task in matrix["base_tasks"]:
        if args.suites and task["suite"] not in args.suites:
            continue
        clean_bddl = task["variants"][next(iter(task["variants"]))][0]["base_bddl"]
        clean_init = task["variants"][next(iter(task["variants"]))][0]["base_init"]
        language = clean_language(task["base_task"])
        for domain, variants in task["variants"].items():
            if args.domains and domain not in args.domains:
                continue
            for v in variants:
                conds.append({"task": task, "domain": domain, "variant": v,
                              "clean_bddl": clean_bddl, "clean_init": clean_init,
                              "language": language, "tag": tag_for(task, domain, v["classification_id"])})
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"refusing to reuse non-empty output dir: {args.out} (pass --resume)")
    args.out.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_meta = {"matrix_sha256": sha(args.matrix), "sweep_script_sha256": sha(Path(__file__)),
                "checkpoint_sha256": sha(args.checkpoint),
                "corp_config_sha256": sha(args.checkpoint.parent / "config.json"),
                "git_rev": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                   stderr=subprocess.DEVNULL, text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"],
                                                          stderr=subprocess.DEVNULL, text=True).strip()),
                "started_at_utc": started, "workers": args.workers, "num_states": args.num_states}
    env = dict(os.environ)
    env.update({"LIBERO_CONFIG_PATH": str(args.libero_config), "MUJOCO_GL": "egl",
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})

    def run_one(c):
        t0 = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cap = args.out / "capture" / c["tag"]
        rep = args.out / "repair" / c["tag"]
        v = c["variant"]; task = c["task"]
        cmd_cap = [sys.executable, str(ROOT / "scripts/v12_capture_features.py"),
                   "--clean-suite", task["suite"], "--clean-task-name", task["base_task"],
                   "--clean-bddl", c["clean_bddl"], "--clean-init-file", c["clean_init"],
                   "--clean-language", c["language"], "--shift-suite", task["suite"],
                   "--shift-task", str(v.get("benchmark_index", v["classification_id"] - 1)), "--shift-kind", c["domain"],
                   "--task-map", str(args.libero_src / "libero/libero/benchmark/libero_suite_task_map.py"),
                   "--task-classification", str(args.libero_src / "libero/libero/benchmark/task_classification.json"),
                   "--num-states", str(args.num_states), "--checkpoint", str(args.checkpoint),
                   "--out", str(cap)]
        log = args.out / "logs" / f"{c['tag']}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as fh:
            r = subprocess.run(cmd_cap, env=env, stdout=fh, stderr=subprocess.STDOUT)
        cap_manifest = cap / "manifest.json"; feat = cap / "features.npz"
        capture_record = {"capture_command": cmd_cap, "capture_returncode": r.returncode,
                          "capture_manifest_sha": sha(cap_manifest), "features_sha": sha(feat),
                          "capture_started_at_utc": t0,
                          "capture_ended_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if r.returncode != 0:
            return {"tag": c["tag"], "stage": "capture", "returncode": r.returncode,
                    "task": task["base_task"], "domain": c["domain"], "cid": v["classification_id"],
                    **capture_record}
        cmd_rep = [sys.executable, str(ROOT / "scripts/e2_resampler_repair_v2.py"),
                   "--dir", str(cap), "--clean-bddl", c["clean_bddl"],
                   "--clean-init-file", c["clean_init"], "--clean-language", c["language"],
                   "--checkpoint", str(args.checkpoint), "--out", str(rep)]
        with log.open("a") as fh:
            r = subprocess.run(cmd_rep, env=env, stdout=fh, stderr=subprocess.STDOUT)
        rep_json = rep / "e2_resampler_repair_v2.json"
        return {"tag": c["tag"], "stage": "repair", "returncode": r.returncode,
                "task": task["base_task"], "domain": c["domain"], "cid": v["classification_id"],
                "capture": str(cap), "repair": str(rep), **capture_record,
                "repair_command": cmd_rep, "repair_returncode": r.returncode,
                "repair_result_sha": sha(rep_json), "log_sha": sha(log),
                "repair_ended_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, c) for c in conds]
        for fut in as_completed(futs):
            try: results.append(fut.result())
            except Exception as exc: results.append({"stage": "exception", "error": repr(exc)})
            print(json.dumps(results[-1]), flush=True)
    run_meta["ended_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (args.out / "sweep_manifest.json").write_text(json.dumps(
        {"run": run_meta, "conditions": conds, "results": results}, indent=2))
    print(json.dumps({"n": len(results), "failures": sum(1 for r in results if r.get("returncode") != 0)}, indent=2))


if __name__ == "__main__":
    main()
