"""Pre-registered task-progress metric for B1 capture data.

p_t = mean over goal literals (On/In obj target) of clip(1 - d_t/d0, 0, 1), where
d_t = Euclidean distance between the body centres of obj_main and target_main and
d0 is the value at the first replan of the episode.  Goal literals are parsed from
the base bddl text stored at capture time.  This rule is fixed before any signal
analysis; it does not use success labels.
"""

from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

CAP = Path("/hy-tmp/validation/b1_capture")
OUT = CAP / "progress.jsonl"
PAT = re.compile(r"\((On|In)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)")


def parse_goal(text):
    return [(a, b, c) for a, b, c in PAT.findall(text)]


def resolve_body(names, raw):
    for cand in (raw + "_main", raw.replace("_contain_region", "") + "_main"):
        if cand in names:
            return names.index(cand)
    return None


def main():
    rows = [json.loads(l) for l in open(CAP / "capture.jsonl") if l.strip()]
    goal_map = json.loads((CAP / "goal_map.json").read_text())
    reps = [r for r in rows if "replan" in r]
    summaries = {(_ := r["_episode_summary"])["task_id"]: _ for r in rows if "_episode_summary" in r}
    by_ep = defaultdict(list)
    for r in reps:
        by_ep[(r["task_id"], r["episode"])].append(r)
    out = []
    for (tid, ep), rs in sorted(by_ep.items()):
        rs.sort(key=lambda x: x["replan"])
        gm = goal_map.get(f"{rs[0]['suite']}|{tid}", {})
        gtext = gm.get("goal_text") or rs[0]["goal_text"]
        literals = parse_goal(gtext)
        if not literals:
            print("NO GOAL", tid, ep)
            continue
        names = rs[0]["body_names"]
        pos0 = np.array([rs[0]["body_pos"][n] for n in names])
        d0s = []
        for lit, obj, tgt in literals:
            oi = resolve_body(names, obj)
            ti = resolve_body(names, tgt)
            if oi is None or ti is None:
                d0s.append(None); continue
            d0s.append(float(np.linalg.norm(pos0[oi] - pos0[ti])))
        prev = -1.0
        for r in rs:
            pos = np.array([r["body_pos"][n] for n in names])
            comps = []
            for (lit, obj, tgt), d0 in zip(literals, d0s):
                if d0 is None or d0 < 1e-9:
                    comps.append(0.0); continue
                oi = resolve_body(names, obj); ti = resolve_body(names, tgt)
                if oi is None or ti is None:
                    comps.append(0.0); continue
                d = float(np.linalg.norm(pos[oi] - pos[ti]))
                comps.append(float(np.clip(1.0 - d / d0, 0.0, 1.0)))
            p = float(np.mean(comps))
            out.append({"task_id": tid, "episode": ep, "replan": r["replan"],
                        "steps": r["steps"], "p": p, "components": comps,
                        "literals": literals, "success_so_far": r["success_so_far"],
                        "monotone_ok": p >= prev - 1e-9})
            prev = p
    (OUT).write_text("\n".join(json.dumps(x) for x in out))
    print("progress rows", len(out))
    for (tid, ep), rs in sorted(by_ep.items()):
        ps = [x["p"] for x in out if x["task_id"] == tid and x["episode"] == ep]
        succ = summaries.get(tid, {}).get("success")
        print(f"task {tid} ep {ep} success={succ} p0={ps[0]:.3f} p_last={ps[-1]:.3f} "
              f"n={len(ps)} monotone_frac={np.mean([x['monotone_ok'] for x in out if x['task_id']==tid and x['episode']==ep]):.2f}")


if __name__ == "__main__":
    main()
