#!/usr/bin/env python3
"""Local-only acceptance bundle; no upload, environment installation or execution.

git bundle preserves the commit and checkout identity required by validation.
The optional Registers source is exported byte-for-byte from its pinned files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dacsc.run_output import output_directory


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pack(out, source):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True
        ).strip()

    if git("status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("commit local changes before packaging")
    analysis = json.loads((ROOT / "analysis_upstream.lock.json").read_text())
    lock = analysis["Registers"]
    # Validate BEFORE creating any archive. This does not import third-party code.
    for rel, expected in lock["files"].items():
        if digest(source / rel) != expected:
            raise ValueError(f"Registers source changed: {rel}")
    with output_directory(out) as out:
        commit = git("rev-parse", "HEAD")
        code = out / "code.bundle"
        subprocess.run(
            ["git", "-C", str(ROOT), "bundle", "create", str(code), "HEAD"], check=True
        )
        subprocess.run(
            ["git", "-C", str(ROOT), "bundle", "verify", str(code)],
            check=True,
            capture_output=True,
        )
        reference = out / "registers_analysis.tgz"
        with tarfile.open(reference, "w:gz") as archive:
            for rel in lock["files"]:
                data = (source / rel).read_bytes()
                if hashlib.sha256(data).hexdigest() != lock["files"][rel]:
                    raise ValueError(f"Registers source changed while packaging: {rel}")
                info = tarfile.TarInfo("registers/" + rel)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                archive.addfile(info, io.BytesIO(data))
            data = (json.dumps(lock, indent=2) + "\n").encode()
            info = tarfile.TarInfo("registers/UPSTREAM.json")
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            archive.addfile(info, io.BytesIO(data))
        runbook = out / "READONLY_SERVER_VALIDATION.md"
        runbook.write_bytes((ROOT / "docs/READONLY_SERVER_VALIDATION.md").read_bytes())
        if git("rev-parse", "HEAD") != commit or git(
            "status", "--porcelain", "--untracked-files=normal"
        ):
            raise RuntimeError(
                "repository changed while packaging; no completed manifest published"
            )
        manifest = {
            "schema": "corp_readonly_bundle_v1",
            "commit": commit,
            "code_bundle_scope": "committed repository history; no ignored datasets/weights/venvs",
            "analysis_source": lock,
            "analysis_contract": analysis,
            "upstream": json.loads((ROOT / "upstream.lock.json").read_text()),
            "files": {p.name: digest(p) for p in (code, reference, runbook)},
            "execution": {
                "server_connection": "not_run",
                "real_video": "not_run",
                "cuda_bf16": "not_run",
                "mujoco_replay": "not_run",
                "training": "not_run",
            },
        }
        (out / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        paths = [*manifest["files"], "bundle_manifest.json"]
        (out / "SHA256SUMS").write_text(
            "".join(f"{digest(out / name)}  {name}\n" for name in paths)
        )
    print(
        json.dumps(
            {
                "commit": commit,
                "bundle": str(out),
                "sha256sums": digest(out / "SHA256SUMS"),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registers-source", type=Path, required=True)
    args = parser.parse_args()
    pack(args.out.expanduser().resolve(), args.registers_source.expanduser().resolve())


if __name__ == "__main__":
    main()
