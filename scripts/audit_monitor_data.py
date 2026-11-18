"""Audit monitor trajectory roles/state grouping and per-step schema."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    split = json.loads(args.split_json.read_text())
    expected = {
        "source_fit": set(split["source_train_state_ids"]),
        "source_validation": set(split["source_val_state_ids"]),
        "target_dev": set(split["target_dev_state_ids"]),
        "target_test": set(split["target_test_state_ids"]),
    }
    report = {"schema": "monitor_data_audit_v2", "roles": {}, "domain_manifests": {}, "errors": []}
    for role in ("source_fit", "source_validation", "target_dev"):
        rdir = args.data_root / role
        files = sorted(rdir.rglob("s*_d*_seed*.json"))
        states = []; domains = Counter(); n_rows = []
        for f in files:
            d = json.loads(f.read_text())
            states.append(int(d["state_id"])); domains[d["domain"]] += 1
            rows = d["rows"]; n_rows.append(len(rows))
            if not rows:
                report["errors"].append(f"empty rows: {f}")
                continue
            for r in rows:
                if len(r["state8"]) != 8 or len(r["clipped_action"]) != 7 or len(r["visual_mean"]) != 768:
                    report["errors"].append(f"bad dims: {f}")
                    break
        report["roles"][role] = {
            "n_episodes": len(files), "states": sorted(set(states)),
            "domains": dict(domains), "n_rows_min": min(n_rows) if n_rows else 0,
            "n_rows_max": max(n_rows) if n_rows else 0,
        }
        if not files:
            report["errors"].append(f"missing role dir: {role}")
            continue
        for domain in ("clean", "shift"):
            mf = rdir / domain / "manifest.json"
            if not mf.exists():
                report["errors"].append(f"missing manifest: {mf}")
                continue
            m = json.loads(mf.read_text())
            report["domain_manifests"].setdefault(role, {})[domain] = {
                "lighting_id": m.get("lighting_id"), "role": m.get("role"),
                "split_sha256": m.get("split_sha256"), "n_records": len(m.get("records", [])),
            }
            if m.get("lighting_id") != 2292:
                report["errors"].append(f"{role}/{domain} lighting {m.get('lighting_id')} != 2292")
            if m.get("role") != role:
                report["errors"].append(f"{role}/{domain} manifest role {m.get('role')}")
        outside = set(states) - expected[role]
        if outside:
            report["errors"].append(f"{role} outside role: {sorted(outside)}")
    for a in ("source_fit", "source_validation", "target_dev"):
        for b in ("source_fit", "source_validation", "target_dev"):
            if a >= b: continue
            overlap = set(report["roles"][a]["states"]) & set(report["roles"][b]["states"])
            if overlap:
                report["errors"].append(f"state overlap {a}/{b}: {sorted(overlap)}")
    target_test = set(report["roles"]["target_dev"]["states"]) & expected["target_test"]
    if target_test:
        report["errors"].append(f"target test states present in target dev: {sorted(target_test)}")
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
