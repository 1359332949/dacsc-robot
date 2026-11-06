#!/usr/bin/env bash
# Local packaging only. Export a clean committed snapshot plus portable code identity.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/.cache/upload}"
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=normal)" ]]; then
  echo "Commit or resolve local changes before packaging; no working-tree/HEAD mixing." >&2
  exit 1
fi
mkdir -p "$OUT"
ARCHIVE="$OUT/dacsc_code.tgz"
if [[ -e "$ARCHIVE" ]]; then
  echo "Refusing to overwrite $ARCHIVE" >&2
  exit 1
fi
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$STAGE"
PYTHONPATH="$ROOT" python3 -m dacsc.code_identity write \
  --root "$STAGE" --commit "$COMMIT" >/dev/null
python3 - "$STAGE" "$ARCHIVE" <<'PY'
import pathlib
import sys
import tarfile

stage = pathlib.Path(sys.argv[1])
archive = pathlib.Path(sys.argv[2])
with tarfile.open(archive, "w:gz") as bundle:
    for path in sorted(stage.rglob("*")):
        bundle.add(path, arcname=path.relative_to(stage), recursive=False)
PY
echo "commit=$COMMIT"
echo "archive=$ARCHIVE"
python3 - "$ARCHIVE" <<'PY'
import hashlib, sys
with open(sys.argv[1], "rb") as stream:
    print("sha256=" + hashlib.file_digest(stream, "sha256").hexdigest())
PY
echo "Code only: includes README, upstream.lock.json, tests and protocol docs."
echo "Fetch dependencies at the locked upstream revisions separately; no upload or deployment was performed."
