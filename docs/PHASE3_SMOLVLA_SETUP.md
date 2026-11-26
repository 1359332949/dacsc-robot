> **2026-11-26 原版评测更新：** 下载并原样运行公开 `roboeval`，使用原始Hub配置：兼容环境十任务Clean **9/10**；另建公开记录主要依赖版本的独立环境，原版单例 **1/1（70步）**。原版视觉编码器BF16，旧自写离线路径FP32，不可视为等价。以下旧桥接数字仅保留历史；不是当前基线。详见 `UPSTREAM_SMOLVLA_REPRODUCTION_V1.md`。

> **2026-11-26 后续诊断已更新：** 真正vanilla+固定社区旧版API的十任务Clean为8/10；LeRobot0.4.4为6/10；官方真vanilla task0成功1/1。旧桥接导致247个GLFW探测进程递归，已按范围修复。Plus基础BDDL/init虽与vanilla相同，但默认纹理分辨率/相机参数不同；不能用桥接后的低分代表vanilla。原始记录保留，以下是旧条件的历史报告；下一步与详细证据见 `SMOLVLA_REPRODUCTION_DIAGNOSIS_V1.md`。尚无第二底座TTA结果。

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

Result: **3/10**, all actions finite. Config B (n_action_steps=10, num_steps=1) gave **1/10**, so the 83% community number did not reproduce.

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


## 5b. Official source difference audit

Checked `lerobot/envs/libero.py` and `lerobot/processor/env_processor.py`:

1. Official `LiberoEnv.step(action)` does **not** remap the gripper or apply
   `[0,1] -> [-1,+1]`; it calls `self._env.step(action)` directly. Our raw
   `OffScreenRenderEnv.step` is therefore the same on this point.
2. Official observation is raw `pixels` (HWC uint8) + nested `robot_state`;
   `preprocess_observation` makes CHW/[0,1] and batch; `LiberoProcessorStep`
   flips H/W. Our path does the same.
3. Remaining differences: our env used 256x256; official `LiberoEnv` config
   defaults to 360x360 with `control_mode="relative"`; our env did not
   explicitly force `controller.use_delta=True`; our max_steps=300 vs official
   spatial max 280; our success check is the CoRP helper.
4. Config A (n_action_steps=1, num_steps=10): 3/10.
   Config B (n_action_steps=10, num_steps=1): **1/10**, not the community 83%.
   Both all-finite.
5. No action/gripper postprocessing difference was found; the low score is not
   explained by the user-suggested gripper remap alone. The official-env
   end-to-end parity run is still required.

## 6. What was not done

- No SigLIP layer sweep.
- No V1 cross-backbone transfer.
- No method comparison.

The forward hook result is valid; the smoke result is a **negative setup
check**, not a method result.

## 7. Official `lerobot-eval` parity attempt

After creating base-init symlinks for LIBERO-plus names, official eval launched:
`lerobot-eval --env.type=libero --env.task=libero_spatial --env.task_ids=[0]`.

Result: **0/1**, 280 steps, 204 s. The task ID 0 in LIBERO-plus is
`..._table_1` (a perturbation), not the vanilla LIBERO base task; the symlinked
base init state does not make it the same benchmark condition. Therefore the
official eval CLI cannot be used directly on this LIBERO-plus asset tree.

This does **not** invalidate the hook result; it confirms the asset/benchmark
mismatch. To compare with the community 83% one must install vanilla LIBERO (or
write a base-task bridge that preserves vanilla task IDs), not just symlink
init files.

## 8. Official vanilla-task bridge result

A `sitecustomize.py` bridge replaced LIBERO-plus `libero_spatial` task map with
the 10 vanilla base tasks and their base init files. Official `lerobot-eval`
then ran task 0 end-to-end:

```text
policy: HuggingFaceVLA/smolvla_libero
env: LiberoEnv, control_mode=relative, 256x256, 280 steps
task 0: 0/1
eval time: 189.9 s
```

This is still below any acceptable Clean baseline. Together with the custom-env
results:

```text
custom E=1,  num_steps=10: 3/10
custom E=10, num_steps=1:  1/10
custom E=8:                4/10
official vanilla task 0:  0/1
```

The official eval infrastructure now runs, but the checkpoint does not reproduce
the community LIBERO-Spatial Clean claim in this environment. Do **not** start
SigLIP layer sweep or V1 cross-backbone experiments.

## 9. Latest experimental conclusions (Phase 3)

- SmolVLA weight integrity and SigLIP hook contract are verified.
- The blocker is the eval/checkpoint protocol, not a missing layer.
- Hand-written and official-vanilla paths both fail Clean parity.
- The only defensible next step is to reproduce the exact community eval asset
  set (vanilla LIBERO + its init files), not to tune the method on a broken
  baseline.
