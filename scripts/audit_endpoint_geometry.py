"""Endpoint audit: On/In geometric progress vs recorded episode success (offline, no rollout)."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dacsc.b1_progress_geometry import p_in, p_on, quat_to_R

CAP = Path("/tmp/b1v2")


def resolve(names, raw):
    for c in (raw + "_main", raw.replace("_contain_region", "") + "_main"):
        if c in names: return names.index(c)
    return None


def target_for(suite, instr):
    if suite == "libero_spatial": return "akita_black_bowl_1", "plate_1", "On"
    for o in ("alphabet_soup_1", "salad_dressing_1", "butter_1"):
        if o.split("_")[0] in instr.lower(): return o, "basket_1", "In"
    return "alphabet_soup_1", "basket_1", "In"


def geom_score(r, kind, oi, ti):
    names = r["body_names"]; bp = r["body_pos"]; bq = r["body_quat"]
    sz = None; tz = None
    for i, b in enumerate(r["geom_bodyid"]):
        if b == oi and (sz is None or np.prod(r["geom_size"][i]) > np.prod(sz)): sz = np.array(r["geom_size"][i])
        if b == ti and (tz is None or np.prod(r["geom_size"][i]) > np.prod(tz)): tz = np.array(r["geom_size"][i])
    if sz is None or tz is None: return None
    op, tp = np.array(bp[names[oi]]), np.array(bp[names[ti]])
    oR, tR = quat_to_R(bq[names[oi]]), quat_to_R(bq[names[ti]])
    return float(p_on(op, sz, oR, tp, tz, tR) if kind == "On" else p_in(op, sz, oR, tp, tz, tR))


def auc(score, label):
    pos = [s for s, y in zip(score, label) if y]; neg = [s for s, y in zip(score, label) if not y]
    if not pos or not neg: return None
    return float(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)))


def main():
    rows = [json.loads(l) for l in open(CAP / "capture.jsonl") if l.strip()]
    reps = [r for r in rows if "replan" in r]
    sums = [r["_episode_summary"] for r in rows if "_episode_summary" in r]
    success = {(s["suite"], s["task_id"], s["episode"]): bool(s["success"]) for s in sums}
    by_ep = defaultdict(list)
    for r in reps: by_ep[(r["suite"], r["task_id"], r["episode"])].append(r)
    out = []
    for key, rs in sorted(by_ep.items()):
        suite, tid, ep = key
        rs.sort(key=lambda x: x["replan"])
        r = rs[0]; names = r["body_names"]
        obj, tgt, kind = target_for(suite, r["instruction"])
        oi, ti = resolve(names, obj), resolve(names, tgt)
        if oi is None or ti is None:
            out.append({"suite": suite, "task_id": tid, "episode": ep, "error": "unresolved"})
            continue
        vals = [geom_score(x, kind, oi, ti) for x in rs]
        vals = [v for v in vals if v is not None]
        # proximity baseline (old definition), d_ref from first replan
        bp0 = np.array(rs[0]["body_pos"][names[oi]]); bt0 = np.array(rs[0]["body_pos"][names[ti]])
        d0 = float(np.linalg.norm(bp0 - bt0))
        prox = []
        for x in rs:
            d = float(np.linalg.norm(np.array(x["body_pos"][names[oi]]) - np.array(x["body_pos"][names[ti]])))
            prox.append(1.0 - d / max(d0, 1e-9))
        out.append({"suite": suite, "task_id": tid, "episode": ep, "kind": kind,
                    "success": success[key], "n_replans": len(rs),
                    "geom_final": float(vals[-1]), "geom_last3_max": float(np.max(vals[-3:])),
                    "geom_max": float(np.max(vals)), "prox_final": float(prox[-1]),
                    "prox_max": float(np.max(prox))})
    ok = [r for r in out if "error" not in r]
    res = {"n_episodes": len(ok), "per_episode": out}
    for metric in ("geom_final", "geom_last3_max", "prox_final"):
        y = [r["success"] for r in ok]
        x = [r[metric] for r in ok]
        succ = [xi for xi, yi in zip(x, y) if yi]
        fail = [xi for xi, yi in zip(x, y) if not yi]
        res[metric] = {"auc_diagnostic": auc(x, y),
                       "median_success": float(np.median(succ)) if succ else None,
                       "median_failure": float(np.median(fail)) if fail else None,
                       "n_success": len(succ), "n_failure": len(fail)}
        for thr in (0.5, 0.8):
            pred = [xi >= thr for xi in x]
            tp = sum(1 for p, yi in zip(pred, y) if p and yi); fp = sum(1 for p, yi in zip(pred, y) if p and not yi)
            fn = sum(1 for p, yi in zip(pred, y) if not p and yi); tn = sum(1 for p, yi in zip(pred, y) if not p and not yi)
            res.setdefault("classification", {}).setdefault(metric, {})[str(thr)] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    Path("/tmp/endpoint_audit.json").write_text(json.dumps(res, indent=2))
    print("episodes", len(ok), "success", sum(1 for r in ok if r["success"]))
    for metric in ("geom_final", "geom_last3_max", "prox_final"):
        print(metric, res[metric])
    for r in ok:
        print(r["suite"], r["task_id"], "ep", r["episode"], "success", r["success"],
              "geom_final", round(r["geom_final"], 3), "geom_max", round(r["geom_max"], 3), "prox", round(r["prox_final"], 3))


if __name__ == "__main__":
    main()
