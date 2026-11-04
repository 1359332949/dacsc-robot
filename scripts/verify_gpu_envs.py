#!/usr/bin/env python3
"""Verify the two isolated GPU environments without training or simulation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_KEYS = ("python", "platform", "torch", "torchvision", "cuda_runtime", "cudnn")


def probe(python: str, profile: str) -> dict:
    profile_imports = (
        "import corp\n"
        if profile == "CoRP"
        else (
            "import robomimic\n"
            "import diffusion_policy\n"
            "import diffusion_policy.dataset.lerobot_dataset\n"
            "import robocasa\n"
            "import robosuite\n"
            "import mujoco\n"
        )
    )
    code = r"""
import json, platform, sys
import torch, torchvision
import numpy
""" + profile_imports + r"""
result = {
    'executable': sys.executable,
    'python': platform.python_version(),
    'platform': platform.platform(),
    'torch': str(torch.__version__),
    'torchvision': str(torchvision.__version__),
    'cuda_runtime': torch.version.cuda,
    'cudnn': torch.backends.cudnn.version(),
    'numpy': str(numpy.__version__),
    'cuda_available': bool(torch.cuda.is_available()),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'capability': list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    'native_bf16': bool(torch.cuda.is_bf16_supported(including_emulation=False)) if torch.cuda.is_available() else False,
}
if 'robosuite' in globals():
    result['robosuite'] = str(robosuite.__version__)
    result['mujoco'] = str(mujoco.__version__)
if torch.cuda.is_available():
    x = torch.ones((2, 2), device='cuda')
    result['cuda_probe'] = float((x @ x).sum().item())
else:
    result['cuda_probe'] = None
print(json.dumps(result))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        [
            str(ROOT),
            str(ROOT / "repos/CoRP"),
            str(ROOT / "repos/robocasa"),
            str(ROOT / "repos/diffusion_policy"),
            str(ROOT / "repos/robomimic"),
            env.get("PYTHONPATH", ""),
        ]
    )
    result = subprocess.run(
        [python, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{profile} import probe failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> int:
    corp_python = os.environ.get("CORP_PYTHON", "")
    robocasa_python = os.environ.get("ROBOCASA_PYTHON", "")
    if not corp_python or not robocasa_python:
        raise SystemExit("CORP_PYTHON and ROBOCASA_PYTHON are required")
    corp = probe(corp_python, "CoRP")
    robocasa = probe(robocasa_python, "RoboCasa")
    checks = {
        "corp_numpy_1_26_4": corp["numpy"] == "1.26.4",
        "robocasa_numpy_2_2_5": robocasa["numpy"] == "2.2.5",
        "torch_2_11_0": corp["torch"].split("+")[0] == robocasa["torch"].split("+")[0] == "2.11.0",
        "cuda_build_12_8": corp["cuda_runtime"] == robocasa["cuda_runtime"] == "12.8",
        "cuda_available": corp["cuda_available"] and robocasa["cuda_available"],
        "native_bf16": corp["native_bf16"] and robocasa["native_bf16"],
        "runtime_signature_equal": all(corp[k] == robocasa[k] for k in RUNTIME_KEYS),
    }
    result = {
        "schema": "corp_gpu_env_acceptance_v1",
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
        "environments": {"corp": corp, "robocasa": robocasa},
        "training_started": False,
        "simulation_started": False,
    }
    out = os.environ.get("VERIFY_OUT", "")
    if out:
        path = Path(out).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
