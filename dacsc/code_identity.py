"""Portable source identity for Git archives without a .git directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "dacsc_code_identity_v1"
IDENTITY_FILE = "CODE_IDENTITY.json"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_commit(commit: str) -> None:
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit)
    ):
        raise ValueError("code identity requires a full lowercase git commit")


def write_package_identity(root: str | Path, commit: str) -> Path:
    root = Path(root).resolve()
    _validate_commit(commit)
    files = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.name != IDENTITY_FILE
        and ".git" not in path.relative_to(root).parts
    }
    if not files:
        raise ValueError("refusing to write an empty code identity")
    payload = {
        "schema": SCHEMA,
        "git_commit": commit,
        "git_dirty": False,
        "files": files,
    }
    path = root / IDENTITY_FILE
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def verify_package_identity(root: str | Path) -> dict:
    root = Path(root).resolve()
    path = root / IDENTITY_FILE
    if not path.is_file():
        raise ValueError("source checkout has no .git or CODE_IDENTITY.json")
    payload = json.loads(path.read_text())
    if set(payload) != {"schema", "git_commit", "git_dirty", "files"}:
        raise ValueError("invalid code identity manifest")
    if payload["schema"] != SCHEMA or payload["git_dirty"] is not False:
        raise ValueError("code identity manifest is not a clean archive")
    _validate_commit(payload["git_commit"])
    files = payload["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("code identity manifest has no files")
    for relative, expected in files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise ValueError("invalid code identity file entry")
        target = root / relative
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"code identity file is missing: {relative}")
        if file_sha256(target) != expected:
            raise ValueError(f"code identity hash mismatch: {relative}")
    return {"git_commit": payload["git_commit"], "git_dirty": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    write = subcommands.add_parser("write")
    write.add_argument("--root", type=Path, required=True)
    write.add_argument("--commit", required=True)
    verify = subcommands.add_parser("verify")
    verify.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "write":
        print(write_package_identity(args.root, args.commit))
    else:
        print(json.dumps(verify_package_identity(args.root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
