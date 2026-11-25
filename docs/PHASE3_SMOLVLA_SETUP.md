# Phase 3 setup: SmolVLA-450M

Status: **environment + checkpoint ready; model NOT run; smoke NOT run.**

## 1. Mirror route (server egress is blocked directly)

| Target | Direct | Working mirror |
|---|---|---|
| HuggingFace | timeout | `https://hf-mirror.com` |
| PyPI | timeout | `https://pypi.tuna.tsinghua.edu.cn/simple` |

The server has no direct internet, but both mirrors are reachable.

## 2. Environment

```text
venv: /hy-tmp/smolvla_venv
python: 3.10.12
lerobot: 0.4.4
torch: 2.10.0+cu128
transformers: 4.57.6
disk free after install + checkpoint: ~26 GB
```

`python3.10-dev` was installed because `evdev` needed `Python.h`.

## 3. Checkpoint

```text
repo_id: HuggingFaceVLA/smolvla_libero
local: /hy-tmp/smolvla_ckpt/HuggingFaceVLA_smolvla_libero
model.safetensors: 1,218,047,032 bytes
sha256: 71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f
```

Downloaded via `hf-mirror` with `aria2c -x16` (direct one-stream download was
~150 KB/s; 16 connections gave ~2.7 MB/s).

Config facts:

```text
input images: observation.images.image, observation.images.image2
resolution: 3 x 256 x 256
state: 8, action: 7
chunk_size: 50
n_action_steps: 1
```

## 4. SigLIP structure (safetensors metadata + lerobot source)

```text
hook path: model.vlm_with_expert.vlm.model.vision_model.encoder.layers.{i}
depth: 12
hidden: 768
patch: 16
input resolution: 512
patches: 1024
CLS token: no (position_embedding is [1024, 768])
layer output: [B, 1024, 768]
25% depth: blocks[2] (0-indexed)
```

Camera handling: `embed_image()` is called once per camera; the two camera
tensors are passed as separate images, not concatenated on the batch dimension.

## 5. What was NOT done

- SmolVLA weights were **not loaded**.
- No inference, no LIBERO smoke, no 10-game Clean run.
- No SigLIP forward hook was executed.

Per instruction: install/download/inspect only; do not run the model yet.

## 6. Next, after confirmation

1. Load checkpoint + run one forward to verify the hook output shape.
2. Confirm observation adapter for `image` / `image2`.
3. Run 10-game libero_spatial Clean smoke at max_steps=300.
4. Only then start SigLIP layer sweep and V1 transfer.

## Files

- `docs/data/phase3/smolvla_smoke.json`
- `scripts/phase3_inspect_smolvla.py` (to be run after confirmation)
