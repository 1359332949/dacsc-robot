#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV:-uv}"
PYPI="${PYPI:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH="${PYTORCH:-https://download.pytorch.org/whl/cu128}"
WHEELHOUSE=""
CORP_PYTHON=""
ROBOCASA_PYTHON=""
ROBOSUITE_SOURCE=""
SKIP_VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
    --corp-python) CORP_PYTHON="$2"; shift 2 ;;
    --robocasa-python) ROBOCASA_PYTHON="$2"; shift 2 ;;
    --robosuite-source) ROBOSUITE_SOURCE="$2"; shift 2 ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || { echo "GPU bootstrap requires Linux" >&2; exit 1; }
[[ -n "$CORP_PYTHON" && -n "$ROBOCASA_PYTHON" ]] || { echo "both --corp-python and --robocasa-python are required" >&2; exit 1; }

if [[ -n "$WHEELHOUSE" ]]; then
  [[ -d "$WHEELHOUSE" ]] || { echo "wheelhouse does not exist: $WHEELHOUSE" >&2; exit 1; }
  if [[ -f "$WHEELHOUSE/SHA256SUMS" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      (cd "$WHEELHOUSE" && sha256sum -c SHA256SUMS)
    else
      (cd "$WHEELHOUSE" && shasum -a 256 -c SHA256SUMS)
    fi
  else
    echo "wheelhouse is missing SHA256SUMS: $WHEELHOUSE" >&2
    exit 1
  fi
  if find "$WHEELHOUSE" -maxdepth 2 \( -name '*.part' -o -name '*.tmp' -o -name '*.chunks' \) -print -quit | grep -q .; then
    echo "wheelhouse contains unfinished downloads: $WHEELHOUSE" >&2
    exit 1
  fi
fi

for python in "$CORP_PYTHON" "$ROBOCASA_PYTHON"; do
  if [[ ! -x "$python" ]]; then
    env_dir="$(dirname "$(dirname "$python")")"
    command -v "$UV" >/dev/null 2>&1 || {
      echo "uv is required to create Python 3.12 environments: $UV" >&2
      exit 1
    }
    "$UV" venv --seed --python 3.12 "$env_dir"
  fi
  "$python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
done

install() {
  local python="$1" lock="$2"
  if [[ -n "$WHEELHOUSE" ]]; then
    "$python" -m pip install \
      --disable-pip-version-check \
      --no-index \
      --find-links "$WHEELHOUSE" \
      --require-hashes \
      -r "$ROOT/environment/$lock"
  else
    "$python" -m pip install \
      --disable-pip-version-check \
      --index-url "$PYPI" \
      --extra-index-url "$PYTORCH" \
      --require-hashes \
      -r "$ROOT/environment/$lock"
  fi
}

install "$CORP_PYTHON" corp.lock.txt
install "$ROBOCASA_PYTHON" robocasa.lock.txt
lerobot_source=()
if [[ -n "$WHEELHOUSE" ]]; then
  lerobot_source=(--no-index --find-links "$WHEELHOUSE")
else
  lerobot_source=(--index-url "$PYPI")
fi
"$ROBOCASA_PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-deps \
  "${lerobot_source[@]}" \
  -r "$ROOT/environment/lerobot.lock.txt"

if [[ -n "$ROBOSUITE_SOURCE" ]]; then
  "$ROBOCASA_PYTHON" -m pip install \
    --disable-pip-version-check \
    --no-build-isolation \
    --no-deps \
    -e "$ROBOSUITE_SOURCE"
elif [[ -n "$WHEELHOUSE" ]]; then
  "$ROBOCASA_PYTHON" -m pip install \
    --disable-pip-version-check \
    --no-index \
    --find-links "$WHEELHOUSE" \
    --no-deps \
    robosuite==1.5.2
else
  "$ROBOCASA_PYTHON" -m pip install \
    --disable-pip-version-check \
    --index-url "$PYPI" \
    --no-deps \
    robosuite==1.5.2
fi

for package in CoRP robomimic diffusion_policy robocasa; do
  [[ -d "$ROOT/repos/$package" ]] || { echo "missing official source: repos/$package" >&2; exit 1; }
done
"$CORP_PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-build-isolation \
  --no-deps \
  -e "$ROOT/repos/CoRP"
"$ROBOCASA_PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-build-isolation \
  --no-deps \
  -e "$ROOT/repos/robomimic" -e "$ROOT/repos/diffusion_policy" -e "$ROOT/repos/robocasa"

if [[ "$SKIP_VERIFY" -eq 1 ]]; then
  echo "Skipped GPU runtime verification (--skip-verify)"
else
  "$ROOT/scripts/verify_gpu_envs.sh" \
    --corp-python "$CORP_PYTHON" \
    --robocasa-python "$ROBOCASA_PYTHON"
fi
