"""Bind a strict phase-A audit to one capture manifest, fail-closed."""

from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.capture / "manifest.json").read_text())
    cond = manifest["condition"]
    cells = json.loads(args.audit.read_text())
    # Audit names are variant names; match domain + variant.
    matches = [c for c in cells
               if c.get("domain") == cond["domain"]
               and c.get("name") == cond["variant_name"]]
    if len(matches) != 1:
        raise SystemExit(f"expected one audit cell for {cond['domain']} {cond['variant_name']}, got {len(matches)}")
    cell = matches[0]
    if not cell.get("all_checks_pass", False):
        raise SystemExit(f"audit cell failed: {cell.get('name')}")
    cap_states = [int(r["index"]) for r in manifest["records"] if r.get("compare_pass")]
    aud_states = [int(s["state"]) for s in cell.get("states", [])]
    if not set(aud_states).issubset(set(cap_states)):
        raise SystemExit(f"audit states not subset of capture states: {aud_states} vs {cap_states}")
    out = {
        "capture_dir": str(args.capture),
        "capture_manifest_sha256": sha(args.capture / "manifest.json"),
        "capture_features_sha256": sha(args.capture / "features.npz"),
        "audit_file": str(args.audit),
        "audit_sha256": sha(args.audit),
        "condition": cond,
        "audit_cell": cell.get("name"),
        "audit_states": aud_states,
        "capture_states": cap_states,
        "all_checks_pass": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"binding": str(args.out), "condition": cond["variant_name"],
                      "all_checks_pass": True}, indent=2))


if __name__ == "__main__":
    main()
