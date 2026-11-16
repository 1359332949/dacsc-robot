"""Run E2 capture + Resampler Oracle diagnostics for every matrix variant.

No training; frozen CoRP.  Parallelizes independent conditions with a process
pool.  One condition = v12 capture then v2 repair diagnostic.
"""

from __future__ import annotations
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tag_for(task, domain, cid):
    safe = task["base_task"].replace("/", "_")
    return f"{task['suite']}__{safe}__{domain}__{cid}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--num-states", type=int, default=50)
    ap.add_argument("--checkpoint", type=Path, default=Path("/hy-tmp/dacsc-robot/repos/CoRP/checkpoints/corp_libero.pt"))
    ap.add_argument("--libero-src", type=Path, default=Path("/hy-tmp/libero_plus"))
    ap.add_argument("--libero-config", type=Path, default=Path("/hy-tmp/libero_config"))
    args = ap.parse_args()
    matrix = json.loads(args.matrix.read_text())
    conds = []
    for task in matrix["base_tasks"]:
        clean_bddl = task["variants"][next(iter(task["variants"]))][0]["base_bddl"]
        clean_init = task["variants"][next(iter(task["variants"]))][0]["base_init"]
        language = " ".join(task["base_task"].split("_"))
        for domain, variants in task["variants"].items():
            if args.domains and domain not in args.domains:
                continue
            for v in variants:
                conds.append({"task": task, "domain": domain, "variant": v,
                              "clean_bddl": clean_bddl, "clean_init": clean_init,
                              "language": language, "tag": tag_for(task, domain, v["classification_id"])})
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"LIBERO_CONFIG_PATH": str(args.libero_config), "MUJOCO_GL": "egl",
                "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})

    def run_one(c):
        cap = args.out / "capture" / c["tag"]
        rep = args.out / "repair" / c["tag"]
        v = c["variant"]; task = c["task"]
        cmd_cap = [sys.executable, str(ROOT / "scripts/v12_capture_features.py"),
                   "--clean-suite", task["suite"], "--clean-task-name", task["base_task"],
                   "--clean-bddl", c["clean_bddl"], "--clean-init-file", c["clean_init"],
                   "--clean-language", c["language"], "--shift-suite", task["suite"],
                   "--shift-task", str(v["classification_id"] - 1), "--shift-kind", c["domain"],
                   "--task-map", str(args.libero_src / "libero/libero/benchmark/libero_suite_task_map.py"),
                   "--task-classification", str(args.libero_src / "libero/libero/benchmark/task_classification.json"),
                   "--num-states", str(args.num_states), "--out", str(cap)]
        log = args.out / "logs" / f"{c['tag']}.log"; log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as fh:
            r = subprocess.run(cmd_cap, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            return {"tag": c["tag"], "stage": "capture", "returncode": r.returncode,
                    "task": task["base_task"], "domain": c["domain"], "cid": v["classification_id"]}
        cmd_rep = [sys.executable, str(ROOT / "scripts/e2_resampler_repair_v2.py"),
                   "--dir", str(cap), "--clean-bddl", c["clean_bddl"],
                   "--clean-init-file", c["clean_init"], "--clean-language", c["language"],
                   "--checkpoint", str(args.checkpoint), "--out", str(rep)]
        with log.open("a") as fh:
            r = subprocess.run(cmd_rep, env=env, stdout=fh, stderr=subprocess.STDOUT)
        return {"tag": c["tag"], "stage": "repair", "returncode": r.returncode,
                "task": task["base_task"], "domain": c["domain"], "cid": v["classification_id"],
                "capture": str(cap), "repair": str(rep)}

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_one, c) for c in conds]
        for fut in as_completed(futs):
            try: results.append(fut.result())
            except Exception as exc: results.append({"stage": "exception", "error": repr(exc)})
            print(json.dumps(results[-1]), flush=True)
    (args.out / "sweep_manifest.json").write_text(json.dumps({"conditions": conds, "results": results}, indent=2))
    print(json.dumps({"n": len(results), "failures": sum(1 for r in results if r.get("returncode") != 0)}, indent=2))


if __name__ == "__main__":
    main()
