#!/usr/bin/env bash
set -euo pipefail

# Runtime libraries required by OpenCV, robosuite/MuJoCo headless rendering and
# GLFW. NVIDIA's driver/EGL libraries are supplied by the host GPU runtime.
PACKAGES=(
  build-essential
  libegl1
  libgl1
  libglib2.0-0
  libglvnd0
  libgomp1
  libosmesa6
  libsm6
  libx11-6
  libxext6
  libxrender1
)

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "GPU system dependencies require Linux" >&2
  exit 1
fi
if ! command -v apt-get >/dev/null 2>&1; then
  echo "only apt-based Linux is supported; install the equivalent OpenGL, EGL and X11 runtime libraries" >&2
  exit 1
fi
if [[ "$(id -u)" -ne 0 ]]; then
  echo "run this script as root or with sudo" >&2
  exit 1
fi

missing=()
for package in "${PACKAGES[@]}"; do
  if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q 'install ok installed'; then
    missing+=("$package")
  fi
done

if [[ ${#missing[@]} -eq 0 ]]; then
  echo "GPU system dependencies already installed"
else
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${missing[@]}"
fi

ldconfig -p | grep -q 'libGL\.so\.1'
ldconfig -p | grep -q 'libEGL\.so\.1'
echo "GPU system dependencies verified"
