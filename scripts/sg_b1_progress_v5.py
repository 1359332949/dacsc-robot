"""B1 progress v5: language-derived On/In targets + real-geometry On/In progress, H=12."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
from pathlib import Path
import numpy as np

from dacsc.b1_progress_geometry import p_in, p_on, quat_to_R

CAP = Path("/hy-tmp/validation/b1_capture_v2")
TARGETS = {"libero_spatial": [("On", "akita_black_bowl_1", "plate_1")]}
OBJECT_TARGET = {"libero_object": [("In", "alphabet_soup_1", "basket_1"),
                                   ("In", "salad_dressing_1", "basket_1"),
                                   ("In", "butter_1", "basket_1")]}
MARGIN_PROX = 1e-9


def resolve(names, raw):
    for cand in (raw + "_main", raw.replace("_contain_region", "") + "_main"):
        if cand in names:
            return names.index(cand)
    return None


def main():
    rows = [json.loads(l) for l in open(CAP / "capture.jsonl") if l.strip()]
    reps = [r for r in rows if "replan" in r]
    by_ep = defaultdict(list)
    for r in reps:
        by_ep[(r["suite"], r["task_id"], r["episode"])].append(r)
    out = []
    for key, rs in sorted(by_ep.items()):
        suite, tid, ep = key
        rs.sort(key=lambda x: x["replan"])
        if suite == "libero_spatial":
            literals = TARGETS[suite]
        else:
            instr = rs[0]["instruction"].lower()
            literals = [c for c in OBJECT_TARGET[suite]
                        if c[1].replace("_", " ") in instr or c[1].split("_")[0] in instr]
            if not literals:
                raise RuntimeError(f"no target for {key}: {instr}")
        names = rs[0]["body_names"]
        geoms = {"size": rs[0]["geom_size"], "pos": rs[0]["geom_pos"], "bodyid": rs[0]["geom_bodyid"]}
        maps = []
        for kind, obj, tgt in literals:
            oi = resolve(names, obj); ti = resolve(names, tgt)
            if oi is None or ti is None:
                raise RuntimeError(f"unresolved {key}: {obj}->{oi}, {tgt}->{ti}")
            maps.append((kind, oi, ti))
        d0 = []
        for kind, oi, ti in maps:
            d0.append(float(np.linalg.norm(np.array(rs[0]["body_pos"][names[oi]])
                                           - np.array(rs[0]["body_pos"][names[ti]]))))
        eps_rows = []
        for r in rs:
            comp_p, comp_g = [], []
            for (kind, oi, ti), d0i in zip(maps, d0):
                obj_pos = np.array(r["body_pos"][names[oi]]); obj_q = np.array(r["body_quat"][names[oi]])
                tgt_pos = np.array(r["body_pos"][names[ti]]); tgt_q = np.array(r["body_quat"][names[ti]])
                obj_size, tgt_size = None, None
                for i, b in enumerate(r["geom_bodyid"]):
                    if b == oi and (obj_size is None or np.prod(r["geom_size"][i]) > np.prod(obj_size)):
                        obj_size = np.array(r["geom_size"][i])
                    if b == ti and (tgt_size is None or np.prod(r["geom_size"][i]) > np.prod(tgt_size)):
                        tgt_size = np.array(r["geom_size"][i])
                if obj_size is None or tgt_size is None:
                    raise RuntimeError(f"missing geom {key}")
                obj_R, tgt_R = quat_to_R(obj_q), quat_to_R(tgt_q)
                comp_p.append(float(np.clip(1.0 - np.linalg.norm(obj_pos - tgt_pos) / max(d0i, MARGIN_PROX), 0.0, 1.0)))
                if kind == "On":
                    comp_g.append(p_on(obj_pos, obj_size, obj_R, tgt_pos, tgt_size, tgt_R))
                else:
                    comp_g.append(p_in(obj_pos, obj_size, obj_R, tgt_pos, tgt_size, tgt_R))
            rec = {"domain": r["domain"], "suite": suite, "task_id": tid, "episode": ep,
                   "seed": r["seed"], "replan": r["replan"], "steps": r["steps"],
                   "p_proximity": float(np.mean(comp_p)), "p_geometric": float(np.mean(comp_g)),
                   "success_so_far": r["success_so_far"]}
            eps_rows.append(rec)
        for i, rec in enumerate(eps_rows):
            j = i + 2
            rec["dp12_proximity"] = None if j >= len(eps_rows) else eps_rows[j]["p_proximity"] - rec["p_proximity"]
            rec["dp12_geometric"] = None if j >= len(eps_rows) else eps_rows[j]["p_geometric"] - rec["p_geometric"]
        out.extend(eps_rows)
    (CAP / "progress_v5.jsonl").write_text("\n".join(json.dumps(x) for x in out))
    print("rows", len(out))
    for key in sorted(by_ep):
        rs = [x for x in out if (x["suite"], x["task_id"], x["episode"]) == key]
        print(key, "n", len(rs), "p_geom_last", round(rs[-1]["p_geometric"], 3),
              "p_prox_last", round(rs[-1]["p_proximity"], 3))


if __name__ == "__main__":
    main()
