# Phase 3 setup: SmolVLA-450M

Status: **BLOCKED before install/download.** Phase 3 has not produced a smoke
result. This is an environment/egress blocker, not a method result.

## 1. Phase 2 residual cleanup (done)

- `scripts/analyze_phase2.py`: reads `sub_group` from the trial JSON; McNemar
  no longer multiplied by 2.
- `scripts/phase2_suite_eval.py`: trial rows now include `sub_group`.
- `docs/data/phase2/libero10_insuite_v1.json`: `vs_frozen_spatial` corrected to
  the actual Spatial-V1 vs Frozen counts.
- Committed in `822cafd`.

## 2. Server preflight

```text
disk /hy-tmp: 35G free
GPU: RTX 3090, 24 GB
system python: 3.11.12
corp venv python: 3.10
```

Network:

```text
huggingface.co:443  -> timeout
pypi.org:443        -> timeout
github.com:443      -> timeout
```

Installed in `corp_venv`:

```text
torch 2.5.1+cu121
transformers 4.44.2
safetensors 0.8.0
huggingface_hub 0.24.6
```

Missing: `lerobot`, `accelerate`, `torchvision`, `diffusers`, `datasets`,
`draccus`, `av`, and the other smolvla deps.

## 3. Checkpoint candidates (HF metadata only, not downloaded)

| repo_id | model size | image keys | resolution | state/action | chunk | n_action_steps | VLM |
|---|---:|---|---|---:|---:|---:|---|
| `HuggingFaceVLA/smolvla_libero` | 1.218 GB | `image`, `image2` | 256 | 8 / 7 | 50 | 1 | SmolVLM2-500M-Instruct |
| `lerobot/smolvla_libero` | 0.907 GB | `camera1`, `camera2` | 256 | 8 / 7 | 50 | 50 | SmolVLM2-500M-Video-Instruct |

The `HuggingFaceVLA` checkpoint has the LIBERO eval metadata and
`n_action_steps=1`; the `lerobot` checkpoint uses `n_action_steps=50`.

## 4. SigLIP structure (partial)

From the SmolVLM2-500M-Video config:

```text
hidden_size = 768
image_size = 512
patch_size = 16
num_attention_heads = 12
depth = not exposed in config
```

The exact number of vision layers and hook path must be read from the loaded
`lerobot`/`transformers` module; it cannot be honestly inferred from config
alone. If depth is 12, 25% depth is `blocks[2]`; if 24, `blocks[5]`.

## 5. Why no smoke was run

The 10-game Clean smoke cannot be executed without:

1. an isolated `smolvla_venv` with `lerobot` and deps;
2. the model weights;
3. a working LIBERO observation adapter for SmolVLA's two camera names.

None of these can be installed because the server has no outbound network.

## 6. Required unblock (choose one)

**A. Enable server egress** to `huggingface.co` + `pypi.org` (or an internal
mirror), then rerun this setup.

**B. Offline bundle transfer** from the Mac host:

```text
host: download lerobot wheel + linux cp310 dependencies + one checkpoint
      (~1.5-2 GB total)
scp to /hy-tmp
server: python -m venv /hy-tmp/smolvla_venv
        pip install --no-index --find-links /hy-tmp/wheels lerobot[...]
```

Disk has 35 GB; the transfer is feasible, but it needs explicit approval to
prepare the offline bundle.

## Decision

Do not start SigLIP layer sweep or V1 transfer until the environment and
checkpoint are actually installed and the 10-game smoke is clean. The current
Phase 3 status is **blocked, no smoke result**.
