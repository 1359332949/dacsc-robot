"""Verify pinned upstream entry sources, without importing simulators or networking."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_upstream(names=None) -> dict:
    raw = (ROOT / "upstream.lock.json").read_bytes()
    lock = json.loads(raw)
    for name in lock if names is None else names:
        base = (
            Path(os.environ["CORP_REPO"])
            if name == "CoRP" and os.environ.get("CORP_REPO")
            else ROOT / "repos" / name
        )
        for path, expected in lock[name]["files"].items():
            actual = hashlib.sha256((base / path).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"Upstream source changed: {name}/{path}; restore revision {lock[name]['revision']}"
                )
    return {
        "lock_sha256": hashlib.sha256(raw).hexdigest(),
        "revisions": {name: item["revision"] for name, item in lock.items()},
    }


def verify_release_checkpoint(path: Path) -> str:
    """Match the released SHA256SUMS before using the LIBERO initialization."""
    expected = json.loads((ROOT / "upstream.lock.json").read_text())["CoRP"][
        "checkpoint_sha256"
    ]
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(
            "official CoRP checkpoint checksum mismatch; restore the released weights"
        )
    return actual


if __name__ == "__main__":
    print(json.dumps(verify_upstream(), indent=2))
