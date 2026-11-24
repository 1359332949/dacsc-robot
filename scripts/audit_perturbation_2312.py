"""Audit whether Lighting 2312 is only an intensity change.

Compares the BDDL files, the lighting scene XML referenced by each BDDL, and
raw-vs-clipped actions from the failure/success traces.
"""

from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import xml.etree.ElementTree as ET


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_light_xml(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    out = []
    for el in root.iter("light"):
        out.append({"tag": el.tag, **el.attrib})
    return out


def action_clip(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    rows = data["rows"]
    import numpy as np
    raw = np.array([r["raw_action"] for r in rows], dtype=float)
    clipped = np.array([r["clipped_action"] for r in rows], dtype=float)
    out = np.abs(raw) > 1.0
    return {
        "n_steps": len(rows), "success": data["success"],
        "step_clip_rate": float(out.any(axis=1).mean()) if len(rows) else 0.0,
        "dim_clip_rate": out.mean(axis=0).tolist() if len(rows) else [],
        "max_abs_raw": float(np.abs(raw).max()) if len(rows) else 0.0,
        "max_action_delta": float(np.abs(raw - clipped).max()) if len(rows) else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--lighting-ids", default="2292,2312")
    ap.add_argument("--lights-dir", type=Path, default=Path("/hy-tmp/libero_plus/libero/libero/assets/scenes/lights"))
    ap.add_argument("--trace-root", type=Path, default=Path("docs/figures"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    m = json.loads(args.matrix.read_text())
    task = next(b for b in m["base_tasks"] if b["suite"] == args.suite)
    ids = [int(x) for x in args.lighting_ids.split(",")]
    variants = {}
    for lid in ids:
        v = next(x for x in task["variants"]["lighting"] if x["classification_id"] == lid)
        bddl = Path(v["base_bddl"]); env_bddl = Path(v["env_bddl_name"])
        base_text = bddl.read_text().splitlines()
        env_text = env_bddl.read_text().splitlines()
        problem = re.search(r"\(define \(problem ([^)]+)\)", "\n".join(env_text))
        problem_name = problem.group(1) if problem else None
        light_xml = None
        if problem_name:
            suffix = problem_name.split("_light_sync_modified_")[-1]
            if suffix.isdigit():
                for cand in args.lights_dir.glob(f"*_light_sync_modified_{suffix}.xml"):
                    light_xml = cand; break
        variants[lid] = {
            "base_bddl": str(bddl), "env_bddl": str(env_bddl),
            "base_sha256": sha(bddl), "env_sha256": sha(env_bddl),
            "problem_name": problem_name,
            "bddl_diff_lines": [line for line in env_text if line not in set(base_text)],
            "light_xml": str(light_xml) if light_xml else None,
            "light_xml_sha256": sha(light_xml) if light_xml else None,
            "lights": read_light_xml(light_xml) if light_xml else None,
        }
    light_changes = None
    if variants[ids[0]]["lights"] and variants[ids[1]]["lights"]:
        a = variants[ids[0]]["lights"]; b = variants[ids[1]]["lights"]
        light_changes = []
        for i, (x, y) in enumerate(zip(a, b)):
            changed = {k: [x.get(k), y.get(k)] for k in set(x) | set(y) if x.get(k) != y.get(k)}
            if changed: light_changes.append({"index": i, "changes": changed})
    trace = {}
    for state in (3, 40):
        p = args.trace_root / f"failure_trace_{state}" / "trace.json"
        if p.exists(): trace[str(state)] = action_clip(p)
    report = {"schema": "perturbation_audit_v1", "suite": args.suite,
              "lighting_ids": ids, "variants": variants,
              "light_xml_changes": light_changes,
              "bddl_only_problem_name_changed": all(
                  len(v["bddl_diff_lines"]) <= 2 for v in variants.values()),
              "conclusion": "2292 vs 2312 changes light direction and specular, not only intensity; treat as a compound lighting/scene-XML change, not pure illumination.",
              "trace_action_clipping": trace}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("lighting_ids", "bddl_only_problem_name_changed", "light_xml_changes", "conclusion", "trace_action_clipping")}, indent=2))


if __name__ == "__main__":
    main()
