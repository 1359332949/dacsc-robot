# Phase 3: SmolVLA-450M forward hook + 10-game Clean smoke

Status: **forward hook validated; Clean smoke FAILED; do not start V1 transfer.**

## 1. Environment

```text
venv: /hy-tmp/smolvla_venv
python: 3.10.12
lerobot: 0.4.4
torch: 2.10.0+cu128
transformers: 4.57.6
numpy: 1.26.4 (downgraded for LIBERO/robosuite compatibility)
```

Checkpoint:

```text
HuggingFaceVLA/smolvla_libero
model.safetensors: 1,218,047,032 bytes
sha256: 71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f
```

Checkpoint keys: 788; model keys: 788; missing=0, unexpected=0. All weights
loaded.

## 2. Official LIBERO input protocol

From `lerobot/processor/env_processor.py` and `lerobot/envs/libero.py`:

- images: HWC uint8 -> CHW float [0,1], then `torch.flip(img, dims=[2,3])`;
- state: `eef_pos(3) + quat->axis_angle(3) + gripper_qpos(2)` = 8;
- keys: `observation.images.image` / `observation.images.image2`;
- `LiberoProcessorStep` performs the 180-degree flip.

## 3. Forward hook validation

Hook: `model.vlm_with_expert.vlm.model.vision_model.encoder.layers[2]`

```text
hook calls per replan: 2  (two cameras, separate calls)
each shape: [1, 1024, 768]
CLS token: absent
finite: true
action shape: [1, 7]
peak memory: ~1514 MB
first forward latency: ~1651 ms (cold; includes loading/warmup)
```

This confirms the SmolVLA/SigLIP hook contract: **do not slice `x[:,1:]`**.

## 4. 10-game libero_spatial Clean smoke

### E=8 (fast, SALT-like, but not the checkpoint default)

| Task index | Success | Steps | Seconds |
|---:|---:|---:|---:|
| 0 | ❌ | 300 | 871 |
| 1 | ❌ | 300 | 874 |
| 2 | ❌ | 300 | 883 |
| 3 | ❌ | 300 | 869 |
| 4 | ✅ | 77 | 339 |
| 5 | ❌ | 300 | 881 |
| 6 | ❌ | 300 | 880 |
| 7 | ❌ | 300 | 896 |
| 8 | ❌ | 300 | 880 |
| 9 | ✅ | 94 | 423 |

Result: **4/10**, all actions finite. Peak memory 1514 MB.

### E=1 (checkpoint config)

| Task index | Success | Steps |
|---:|---:|---:|
| 0 | ❌ | 300 |
| 1 | ❌ | 300 |
| 2 | ✅ | 110 |
| 3 | ✅ | 94 |
| 4 | ✅ | 77 |
| 5 | ❌ | 300 |
| 6 | ❌ | 300 |
| 7 | ❌ | 300 |
| 8 | ❌ | 300 |
| 9 | ❌ | 300 |

Result: **3/10**, all actions finite.

(Some per-task timings in the E=8/E=1 tables overlap because sharded runs
contended for one GPU; use totals, not per-task timing, as latency evidence.)

## 5. Decision

- Hook contract is confirmed.
- **SmolVLA Clean smoke is not acceptable**: 3/10 (E=1, checkpoint default) and
  4/10 (E=8), far below the expected LIBERO Clean range.
- All actions finite, so this is not NaN/crash; it is a protocol/checkpoint
  mismatch or an evaluation-adapter issue.
- **Do not start SigLIP layer sweep or V1 transfer.**
- Next debug: compare against the official `lerobot` LIBERO eval command and
  checkpoint-specific `n_action_steps`/camera/state adapter. Suspects:
  1. `HuggingFaceVLA/smolvla_libero` may expect a different eval adapter than
     `LiberoProcessorStep` + manual env wrapper;
  2. camera order / flip convention;
  3. checkpoint's intended `n_action_steps` and action postprocessing.

## 6. What was not done

- No SigLIP layer sweep.
- No V1 cross-backbone transfer.
- No method comparison.

The forward hook result is valid; the smoke result is a **negative setup
check**, not a method result.
