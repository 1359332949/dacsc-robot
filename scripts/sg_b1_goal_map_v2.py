"""Goal text map for capture v2 (longest-prefix base bddl match)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "repos/CoRP/scripts"))
from evaluate_libero import configure_libero
class C: pass
c = C(); c.libero_src = Path("/hy-tmp/libero_plus"); c.libero_config_path = Path("/hy-tmp/libero_plus_config")
c.render_gpu = 0; c.device = "cuda:0"
configure_libero(c)
from libero.libero import benchmark, get_libero_path
CAP = Path("/hy-tmp/validation/b1_capture_v2")
rows = [json.loads(l) for l in open(CAP / "capture.jsonl") if l.strip()]
keys = sorted({(r["suite"], r["task_id"]) for r in rows if "replan" in r})
suites = {s: benchmark.get_benchmark_dict()[s]() for s in ("libero_spatial", "libero_object")}
out = {}
for suite, tid in keys:
    task = suites[suite].get_task(tid)
    folder = Path(get_libero_path("bddl_files")) / task.problem_folder
    import re as _re
    cand_names = []
    for pat in (r"_(table|tb)_\d+$", r"_view_.*$", r"_add_\d+$",
                r"_(light|noise|background).*$"):
        base = _re.sub(pat, "", task.name)
        if base != task.name:
            cand_names.append(base)
    found = None
    for base in cand_names:
        cand = folder / (base + ".bddl")
        if cand.exists():
            found = cand; break
    if found is None:
        parts = task.name.split("_")
        for k in range(len(parts), 0, -1):
            cand = folder / ("_".join(parts[:k]) + ".bddl")
            if cand.exists(): found = cand; break
    out[f"{suite}|{tid}"] = {"task_name": task.name, "base_bddl": str(found) if found else None,
                             "goal_text": found.read_text() if found else ""}
    print(suite, tid, "->", found)
(CAP / "goal_map.json").write_text(json.dumps(out, indent=2))
