"""B2 offline wrap-up: record counts, candidate paired contrasts, censor-consistency checks."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

SRC = Path("/hy-tmp/validation/b2v3_formal/b2_v3.jsonl")
OUT = Path("/hy-tmp/validation/b2v3_formal/offline.json")


def ci90(vals, groups, n=1000, seed=0):
    keys = sorted(set(groups)); by = {k: [v for v, g in zip(vals, groups) if g == k] for k in keys}
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        out.append(float(np.mean([v for k in pick for v in by[k]])))
    return [float(np.percentile(out, 5)), float(np.percentile(out, 95))]


def main():
    rows = [json.loads(l) for l in open(SRC)]
    cand = [r for r in rows if "candidate" in r]
    skips = [r for r in rows if r.get("censor") == "stage_or_success"]
    complete = [r for r in cand if "tau_prox" in r and "censor" not in r]
    res = {"n_jsonl": len(rows), "n_candidate_records": len(cand), "n_stage_skips": len(skips),
           "n_complete": len(complete), "n_censored": len(cand) - len(complete),
           "by_candidate": {}}
    for c in sorted({r["candidate"] for r in cand}):
        sel = [r for r in cand if r["candidate"] == c]
        comp = [r for r in sel if "tau_prox" in r and "censor" not in r]
        res["by_candidate"][c] = {"records": len(sel), "complete": len(comp),
                                  "censored": len(sel) - len(comp),
                                  "censor_reasons": dict(Counter(r["censor"] for r in sel if "censor" in r))}
    # paired contrasts on identical (suite,task,episode,branch)
    tau = {}
    for r in complete:
        key = (r["suite"], r["task_id"], r["episode"], r["branch"])
        tau.setdefault(key, {})[r["candidate"]] = r["tau_prox"]
    pairs = [("raw_proj_100", "raw_full_100"), ("raw_proj_100", "raw_sens_100"),
             ("raw_full_100", "raw_sens_100"), ("raw_sens_100", "prod_step_005"),
             ("raw_proj_100", "prod_step_005"), ("raw_full_100", "prod_step_005")]
    res["paired_contrasts"] = {}
    for a, b in pairs:
        keys = [k for k, d in tau.items() if a in d and b in d]
        if not keys:
            continue
        d = [tau[k][a] - tau[k][b] for k in keys]
        groups = [(k[0], k[1]) for k in keys]
        res["paired_contrasts"][f"{a}_minus_{b}"] = {
            "n_pairs": len(d), "mean_delta": float(np.mean(d)),
            "median_delta": float(np.median(d)), "ci90": ci90(d, groups)}
    # censor consistency: done_* true but success_* false, or unequal steps
    conflicts = []
    for r in cand:
        if "censor" not in r: continue
        dc, du = bool(r.get("done_control")), bool(r.get("done_update"))
        sc, su = bool(r.get("success_control")), bool(r.get("success_update"))
        steps_unequal = r.get("steps_control") != r.get("steps_update")
        if (dc and not sc) or (du and not su) or (sc != su and steps_unequal):
            conflicts.append({"candidate": r.get("candidate"), "domain": r.get("domain"),
                              "suite": r.get("suite"), "task_id": r.get("task_id"),
                              "episode": r.get("episode"), "branch": r.get("branch"),
                              "censor": r.get("censor"), "steps_control": r.get("steps_control"),
                              "steps_update": r.get("steps_update"),
                              "success_control": sc, "success_update": su,
                              "done_control": dc, "done_update": du})
    res["censor_conflicts"] = conflicts
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("n_jsonl", "n_candidate_records", "n_stage_skips",
                                          "n_complete", "n_censored")}, indent=2))
    print("paired contrasts", json.dumps(res["paired_contrasts"], indent=2))
    print("censor conflicts", len(conflicts))


if __name__ == "__main__":
    main()
