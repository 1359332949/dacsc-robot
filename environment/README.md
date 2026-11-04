# GPU environment reproducibility

The GPU environment is split because the reviewed upstream constraints conflict:

- CoRP requires `numpy==1.26.4`.
- RoboCasa requires `numpy==2.2.5`.
- Both use the Python 3.12 / Linux x86-64 / PyTorch 2.11.0 CUDA 12.8 stack.

The validation receipt accepts different virtualenv paths only when Python,
platform, Torch, Torchvision, CUDA runtime and cuDNN versions match.

## Local preparation

Build locks for the Linux target without installing anything:

```bash
./scripts/build_gpu_env_locks.sh
```

Download a reusable Linux wheelhouse:

```bash
./scripts/download_gpu_wheelhouse.sh /absolute/path/to/wheelhouse
```

Both commands use PyTorch's CUDA 12.8 index and a China-local Python mirror.
The wheelhouse is architecture-specific; it is not a macOS environment. Large
wheels are fetched as parallel HTTP ranges. Completed chunks are retained, so
re-running the same command resumes an interrupted transfer. Every wheel is
accepted only after its locked SHA256 digest matches.

## Local Linux acceptance

Validate both environments locally without a GPU or a remote server:

```bash
./scripts/verify_gpu_envs_local_docker.sh \
  /absolute/path/to/wheelhouse
```

On Apple Silicon this runs an emulated `linux/amd64` container and is slower
than the target server. It exercises the exact system-library install,
hash-locked offline installation, official editable packages and imports. It
does not test CUDA or native BF16; those still require the target NVIDIA host.

## Server bootstrap

Install the Linux OpenGL/EGL/X11 runtime libraries once:

```bash
sudo ./scripts/install_gpu_system_deps.sh
```

This requires an apt-based Linux host. It installs only missing runtime
libraries used by OpenCV, robosuite/MuJoCo headless rendering and GLFW, plus the
C toolchain required to build `evdev` (a source-only dependency of `pynput`).
It does not install or replace the NVIDIA host driver.

With network access:

```bash
uv python install 3.12
./scripts/bootstrap_gpu_envs.sh \
  --corp-python /hy-tmp/envs/corp/bin/python \
  --robocasa-python /hy-tmp/envs/robocasa/bin/python
```

The two `--*-python` paths may point to missing virtualenvs; the bootstrap
creates each one with Python 3.12 before installing the locked requirements.

With a prepared wheelhouse:

```bash
./scripts/bootstrap_gpu_envs.sh \
  --wheelhouse /hy-tmp/wheelhouse \
  --corp-python /hy-tmp/envs/corp/bin/python \
  --robocasa-python /hy-tmp/envs/robocasa/bin/python
```

The script verifies `SHA256SUMS`, refuses unfinished downloads, non-Linux hosts
and Python versions other than 3.12, then installs only from the wheelhouse.
`uv` is needed only when a requested Python 3.12 environment does not exist;
package installation uses the standard `pip` already present in that
environment. It installs `robosuite` with `--no-deps` because the 1.5.2 wheel
metadata lists the optional `mink` extra as mandatory and pins NumPy below 2.
For MuJoCo replay, provide the pinned robosuite master checkout explicitly; do
not weaken RoboCasa's NumPy assertion.

`lerobot==0.3.3` is installed from its official wheel with `--no-deps` because
its metadata over-constrains Torch to `<2.8` while the verified runtime uses
Torch 2.11. RoboCasa playback imports only its dataset helpers; their actual
runtime dependencies (`datasets`, `jsonlines`) are pinned in the main RoboCasa
lock and covered by smoke tests. No Torch downgrade or source patch is applied.

`tianshou==0.4.10` is not installed: its `protobuf<3.20` requirement conflicts
with the pinned TensorBoard stack. It is used only by RoboCasa's optional speed
benchmark script, not by dataset reading, policy inference or MuJoCo playback.

## Verification

```bash
./scripts/verify_gpu_envs.sh \
  --corp-python /hy-tmp/envs/corp/bin/python \
  --robocasa-python /hy-tmp/envs/robocasa/bin/python \
  --out /hy-tmp/validation/env_acceptance.json
```

The verifier checks versions, CUDA availability, native BF16, imports, and that
both environments expose the same runtime signature. A passing environment
check is a prerequisite for the read-only video/CUDA validation, not a
substitute for it.
