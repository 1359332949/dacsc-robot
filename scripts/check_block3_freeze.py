"""Read-only seal check for this revision. Not a rollout or an authorization bypass."""

import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "protocols/block3_confirmation_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    seal = json.loads((P / "seal.json").read_text())
    lock = json.loads((P / "method_lock.json").read_text())
    audit = json.loads((P / "history_audit.json").read_text())
    smoke = json.loads((P / "object_clean_smoke.json").read_text())
    for name, h in {**lock["files"], **seal["files"]}.items():
        if sha(ROOT / name) != h:
            raise ValueError("changed: " + name)
    for name, h in lock["refs"].items():
        if sha(P / name) != h:
            raise ValueError("reference changed: " + name)
    if audit["status"] != "HISTORY_INCOMPLETE_NO_SCORING" or any(
        r["certified_frozen_only"] for r in audit["factors"].values()
    ):
        raise ValueError("unexpected historical release status")
    if {r["role"] for r in smoke["rows"]} != {
        "compatibility_only_exclude_from_scoring"
    } or smoke["scoring_authorized"]:
        raise ValueError("invalid compatibility role")
    return {
        "method": "FROZEN_NO_EMA",
        "eligibility": audit["status"],
        "object": smoke["status"],
        "official_scoring": "LOCKED",
        "cross_suite_scoring": "LOCKED",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--require-scoring", action="store_true")
    a = p.parse_args()
    print(json.dumps(check()))
    if a.require_scoring:
        raise SystemExit(2)
