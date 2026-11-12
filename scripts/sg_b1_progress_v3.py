"""B1 progress v3: proximity + region-aware geometric On/In progress (strict mapping, H=12)."""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

CAP = Path("/hy-tmp/validation/b1_capture_v2")
PAT = re.compile(r"\((On|In)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)")


def resolve(names, raw):
    for cand in (raw + "_main", raw.replace("_contain_region", "") + "_main"):
        if cand in names:
            return names.index(cand)
    return None


REG = re.compile(r"\((\w+)\s*\(:target\s+([\w]+)\)\s*\(:ranges\s*\(\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)")


def parse_regions(text):
    out = {}
    for name, tgt, x0, y0, x1, y1 in REG.findall(text):
        out[name] = (tgt, float(x0), float(y0), float(x1), float(y1))
    return out


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
        all_lits = [(a, b, c) for a, b, c in PAT.findall(
            gm.get(f"{suite}|{tid}", {}).get("goal_text") or rs[0]["goal_text"])]
        # pre-registered: floor_* literals encode initial placements, not task progress
        literals = [x for x in all_lits if "floor_" not in x[2]]
        if not literals:
            raise RuntimeError(f"no task-progress literal for {key}")
        names = rs[0]["body_names"]
        geoms = (rs[0]["geom_params"], rs[0]["geom_size"], rs[0]["geom_pos"]) if False else None
        # per-episode mapping must be strict
        regions = parse_regions(gm.get(f"{suite}|{tid}", {}).get("goal_text") or rs[0]["goal_text"])
        names0 = rs[0]["body_names"]
        table_idx = names0.index("table") if "table" in names0 else None
        table_pos = np.array(rs[0]["body_pos"]["table"]) if table_idx is not None else None
        table_top = None
        if table_idx is not None:
            gids = [i for i, b in enumerate(rs[0]["geom_bodyid"]) if b == table_idx]
            gbest = max(gids, key=lambda i: float(np.prod(rs[0]["geom_size"][i])))
            table_top = float(rs[0]["geom_pos"][gbest][2] + rs[0]["geom_size"][gbest][2] / 2)
        maps = []
        for kind, obj, tgt in literals:
            oi = resolve(names, obj)
            if oi is None:
                raise RuntimeError(f"unresolved object for {key}: {obj}")
            if tgt in regions and table_pos is not None:
                tn, x0, y0, x1, y1 = regions[tgt]
                center = table_pos + np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0])
                size = np.array([x1 - x0, y1 - y0, 0.05])
                maps.append((kind, oi, center, size, True))
            else:
                ti = resolve(names, tgt)
                if ti is None:
                    raise RuntimeError(f"unresolved target for {key}: {tgt}")
                tgeo = target_geom(names, rs[0]["geom_bodyid"], rs[0]["geom_size"], rs[0]["geom_pos"], ti)
                if tgeo[0] is None:
                    raise RuntimeError(f"no target geom for {key}: {tgt}")
                maps.append((kind, oi, tgeo[0], tgeo[1], False))
        d0 = []
        for kind, oi, tpos, tsize, is_region in maps:
            d0.append(float(np.linalg.norm(np.array(rs[0]["body_pos"][names[oi]]) - np.asarray(tpos))))
        prox, geom = [], []
        for r in rs:
            bp = r["body_pos"]
            comp_p, comp_g = [], []
            for (kind, oi, tpos, tsize, is_region), d0i in zip(maps, d0):
                obj_p = np.array(bp[names[oi]]); tgt_p = np.asarray(tpos)
                comp_p.append(float(np.clip(1.0 - np.linalg.norm(obj_p - tgt_p) / max(d0i, 1e-9), 0.0, 1.0)))
                if is_region and table_top is not None:
                    comp_g.append(float(np.clip(1.0 - abs(obj_p[2] - table_top - 0.03) / 0.08, 0.0, 1.0))
                                 if kind == "On" else
                                 float(np.clip(1.0 - max(0.0, obj_p[2] - table_top - tsize[2]) / max(tsize[2], 1e-3), 0.0, 1.0)))
                else:
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
