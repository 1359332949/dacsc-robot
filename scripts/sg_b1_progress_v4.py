"""B1 progress v4: language-derived On/In targets (fixed before analysis), strict mapping, H=12.

Manual target map (from task language + bddl object names, no success information):
  libero_spatial: akita_black_bowl_1 On plate_1
  libero_object:  alphabet_soup_1 / salad_dressing_1 / butter_1 In basket_1
"""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

CAP = Path("/hy-tmp/validation/b1_capture_v2")
TARGETS = {
    "libero_spatial": [("On", "akita_black_bowl_1", "plate_1")],
}
OBJECT_TARGET = {"libero_object": [("In", "alphabet_soup_1", "basket_1"),
                                   ("In", "salad_dressing_1", "basket_1"),
                                   ("In", "butter_1", "basket_1")]}


def resolve(names, raw):
    for cand in (raw + "_main", raw.replace("_contain_region", "") + "_main"):
        if cand in names:
            return names.index(cand)
    return None


def target_geom(names, geom_bodyid, geom_size, geom_pos, ti):
    idxs = [i for i, b in enumerate(geom_bodyid) if b == ti]
    if not idxs:
        return None, None
    best = max(idxs, key=lambda i: float(np.prod(geom_size[i])))
    return np.array(geom_pos[best]), np.array(geom_size[best])


def p_geometric(kind, obj_p, tpos, tsize):
    dx, dy, dz = obj_p - tpos
    sx, sy, sz = [max(float(v), 1e-3) for v in tsize]
    inside_x = float(np.clip(1.0 - max(abs(dx) - sx / 2, 0.0) / (sx / 2), 0.0, 1.0))
    inside_y = float(np.clip(1.0 - max(abs(dy) - sy / 2, 0.0) / (sy / 2), 0.0, 1.0))
    inside_xy = inside_x * inside_y
    if kind == "On":
        height = float(np.clip(1.0 - abs(dz - (sz / 2 + 0.02)) / 0.06, 0.0, 1.0))
        return inside_xy * height
    depth = float(np.clip(1.0 - max(0.0, dz - sz / 2) / sz, 0.0, 1.0))
    return inside_xy * depth


def main():
    rows = [json.loads(l) for l in open(CAP / "capture.jsonl") if l.strip()]
    gm = json.loads((CAP / "goal_map.json").read_text())
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
            # choose the object mentioned in the instruction
            instr = rs[0]["instruction"].lower()
            literals = [cand for cand in OBJECT_TARGET[suite] if cand[1].replace("_", " ") in instr
                        or cand[1].split("_")[0] in instr]
            if not literals:
                raise RuntimeError(f"no instruction-matched target for {key}: {instr}")
        names = rs[0]["body_names"]
        geoms = (rs[0]["geom_params"], rs[0]["geom_size"], rs[0]["geom_pos"]) if False else None
        # per-episode mapping must be strict
        maps = []
        for kind, obj, tgt in literals:
            oi = resolve(names, obj); ti = resolve(names, tgt)
            if oi is None or ti is None:
                raise RuntimeError(f"unresolved body for {key}: {obj}->{oi}, {tgt}->{ti}")
            tgeo = target_geom(names, rs[0]["geom_bodyid"], rs[0]["geom_size"], rs[0]["geom_pos"], ti)
            if tgeo[0] is None:
                raise RuntimeError(f"no target geom for {key}: {tgt}")
            maps.append((kind, oi, ti, tgeo[0], tgeo[1]))
        d0 = []
        for kind, oi, ti, tpos, tsize in maps:
            d0.append(float(np.linalg.norm(np.array(rs[0]["body_pos"][names[oi]]) - np.array(rs[0]["body_pos"][names[ti]]))))
        prox, geom = [], []
        for r in rs:
            bp = r["body_pos"]
            comp_p, comp_g = [], []
            for (kind, oi, ti, tpos, tsize), d0i in zip(maps, d0):
                obj_p = np.array(bp[names[oi]]); tgt_p = np.array(bp[names[ti]])
                comp_p.append(float(np.clip(1.0 - np.linalg.norm(obj_p - tgt_p) / max(d0i, 1e-9), 0.0, 1.0)))
                comp_g.append(p_geometric(kind, obj_p, tgt_p, tsize))
            rec = {"domain": r["domain"], "suite": suite, "task_id": tid, "episode": ep,
                   "seed": r["seed"], "replan": r["replan"], "steps": r["steps"],
                   "p_proximity": float(np.mean(comp_p)), "p_geometric": float(np.mean(comp_g)),
                   "success_so_far": r["success_so_far"]}
            out.append(rec)
        # compute dp12 by replan index order
        eps_rows = [x for x in out if (x["suite"], x["task_id"], x["episode"]) == key]
        for i, rec in enumerate(eps_rows):
            j = i + 2
            rec["dp12_proximity"] = None if j >= len(eps_rows) else eps_rows[j]["p_proximity"] - rec["p_proximity"]
            rec["dp12_geometric"] = None if j >= len(eps_rows) else eps_rows[j]["p_geometric"] - rec["p_geometric"]
    (CAP / "progress_v2.jsonl").write_text("\n".join(json.dumps(x) for x in out))
    print("rows", len(out))
    # summary per episode
    for key in sorted(by_ep):
        rs = [x for x in out if (x["suite"], x["task_id"], x["episode"]) == key]
        print(key, "n", len(rs), "p_geom_last", round(rs[-1]["p_geometric"], 3),
              "p_prox_last", round(rs[-1]["p_proximity"], 3))


if __name__ == "__main__":
    main()
