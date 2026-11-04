#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELHOUSE="${1:?usage: verify_gpu_envs_local_docker.sh WHEELHOUSE}"
DOCKER="${DOCKER:-docker}"
IMAGE="${IMAGE:-python:3.12-slim}"

[[ -d "$WHEELHOUSE" ]] || { echo "wheelhouse does not exist: $WHEELHOUSE" >&2; exit 1; }
command -v "$DOCKER" >/dev/null 2>&1 || { echo "docker is required" >&2; exit 1; }

"$DOCKER" run --rm --platform linux/amd64 \
  -v "$ROOT:/repo:ro" \
  -v "$(cd "$WHEELHOUSE" && pwd):/wheelhouse:ro" \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    /repo/scripts/install_gpu_system_deps.sh
    python -m venv /tmp/corp
    python -m venv /tmp/robocasa
    /repo/scripts/bootstrap_gpu_envs.sh \
      --wheelhouse /wheelhouse \
      --corp-python /tmp/corp/bin/python \
      --robocasa-python /tmp/robocasa/bin/python \
      --skip-verify
    /tmp/corp/bin/python -c "
import corp
import numpy
import torch
import torchvision
import transformers
print(\"CoRP\", torch.__version__, torchvision.__version__, numpy.__version__, transformers.__version__)
"
    PYTHONPATH=/repo/repos/diffusion_policy /tmp/robocasa/bin/python -c "
import diffusion_policy
import diffusion_policy.dataset.lerobot_dataset
import mujoco
import numpy
import robocasa
import robomimic
import robosuite
import torch
import torchvision
print(\"RoboCasa\", torch.__version__, torchvision.__version__, numpy.__version__, mujoco.__version__, robosuite.__version__)
"
  '
