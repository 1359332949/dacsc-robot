"""Metadata-only triage. Absence of a name/ID in logs never grants admission."""

import argparse, collections, hashlib, json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/data"
GROUPS = ("lighting", "background", "sensor_noise")
REG = "832f77b5e03d8c69e50c49d18fba36739ff7bd2e4f4d10e1501f9d7bb41b00fe"
DEV = "d3e0d8c642cd215c879d2f0174663ff5ec585cd5c6d6b531a213c26e9fb982c0"


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(a):
    assert digest(a.registry) == REG
    data = json.loads(a.registry.read_text())
    rows = [
        r
        for r in data["registry"]
        if r["factor"] in GROUPS and r["suite"] == "libero_spatial"
    ]
    mp = DATA / "static_visual_v2_suite/final/dev_manifest.json"
    assert digest(mp) == DEV
    dev = json.loads(mp.read_text())["cases"]
    byname = {r["logical_task_name"].encode(): r["equivalence_group"] for r in rows}
    byid = {r["case_id"].encode(): r["equivalence_group"] for r in rows}
    reason = collections.defaultdict(set)
    names = re.compile(rb"pick_up_the_black_bowl_[a-zA-Z0-9_]+")
    ids = re.compile(rb"(?<![a-zA-Z0-9])[a-f0-9]{24}(?![a-zA-Z0-9])")
    receipts = []
    for r in dev:
        if r["factor"] in GROUPS:
            reason[r["equivalence_group"]].add("v2_dev")
    for r in rows:
        if r["history_status"] != "candidate" or r["classification_id"] in (2292, 2312):
            reason[r["equivalence_group"]].add("legacy_quarantine")

    def scan(lines, tag):
        seen = set()
        markers = set()
        for line in lines:
            seen.update(byname[v] for v in names.findall(line) if v in byname)
            if b"case_id" in line:
                seen.update(byid[v] for v in ids.findall(line) if v in byid)
            for token in (
                b"topk10_l3_extra3",
                b"debias_cam_obj_big",
                b"base_full=10030/10030",
            ):
                if token in line:
                    markers.add(token.decode())
        for g in seen:
            reason[g].add(tag)
        return dict(groups=len(seen), markers=sorted(markers))

    fs = sorted(a.history.glob("*/session.v4.jsonl.zstd"))
    assert fs
    for f in fs:
        with subprocess.Popen(["zstd", "-dc", str(f)], stdout=subprocess.PIPE) as proc:
            r = scan(proc.stdout, "archive_literal_reference")
            assert proc.wait() == 0
        receipts.append(dict(source=f.parent.name, sha256=digest(f), **r))
    for f in a.extra:
        with f.open("rb") as handle:
            r = scan(handle, "extra_legacy_reference")
        receipts.append(dict(source=f.name, sha256=digest(f), **r))
    # Longer-lived equivalence: exclude same benchmark instance independently of init seed/index.
    key = lambda r: (
        r["base_task"],
        r["resolved_bddl_hash"],
        json.dumps(
            {
                k: v
                for k, v in r["runtime_params"].items()
                if not k.endswith("_asset_id")
            },
            sort_keys=True,
        ),
    )
    unions = collections.defaultdict(set)
    for r in rows:
        unions[key(r)].update(reason[r["equivalence_group"]])
    counts = {}
    for fac in GROUPS:
        sub = {key(r) for r in rows if r["factor"] == fac}
        counts[fac] = dict(
            raw_ids=sum(r["factor"] == fac for r in rows),
            groups=len(sub),
            excluded_or_referenced=sum(bool(unions[k]) for k in sub),
            unresolved=sum(not unions[k] for k in sub),
            certified_frozen_only=0,
        )
    return dict(
        status="HISTORY_INCOMPLETE_NO_SCORING",
        registry_sha256=REG,
        dev_sha256=DEV,
        prior_history_complete=data["settings"]["history_complete"],
        factors=counts,
        log_inventory=receipts,
        excluded_group_ids=sorted(
            {r["equivalence_group"] for r in rows if unions[key(r)]}
        ),
        scope="Literal references (including enumeration) are conservatively excluded, not proof of TTA. Unlogged or index/range-only jobs and use of per-ID Frozen outcomes for curation remain unresolved. No admission list is emitted.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("registry", type=Path)
    p.add_argument("history", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--extra", type=Path, action="append", default=[])
    a = p.parse_args()
    a.out.write_text(json.dumps(run(a), indent=2) + "\n")
