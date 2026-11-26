> **2026-11-26 对照实现审计更新：** 本文旧 `actmad_true` 是自行移植版本，不能当作原版 ActMAD 的忠实复现；其多帧双相机排列存在已确认错误。修复后实现标识为 `actmad_inspired_viewwise_v2`，已通过真实模型执行链检查，但尚未重新取得闭环对比成绩。原数字保留作历史记录，不能据此主张优于原版 ActMAD。后续共享校准阈值另有公式问题，须分别追溯。见 `BASELINE_IMPLEMENTATION_AUDIT_V2.md`。

# Phase 2A pilot: libero_10 Clean, 5 arms

Status: **pilot only; full 400-condition table not yet run.**

## Protocol

10 official libero_10 base tasks, `init_index=1` (one unseen state per task),
EGL, FP32, H=E=12, max_steps=520, `env.noise=0`, shared Spatial calibration.
Five paired arms: frozen, pixel_align, deep_block12, actmad_true, ours_v1.

## Results (10 conditions x 5 arms = 50 rollouts)

| Arm | Clean |
|---|---:|
| Frozen | 9/10 |
| Pixel-Align | 7/10 |
| Deep Block-12 | 9/10 |
| True ActMAD | 9/10 |
| Ours V1 | 9/10 |

All five arms fail the same condition (KITCHEN_SCENE3 init_index=1), which is a
hard task/state, not a method-specific failure. No method separates on this
10-condition pilot.

## Why the full 2,000-rollout table is not reported

The smoke plus pilot show a legitimate hard Clean condition. The full
FairPhysical100x4 table needs 400 conditions x 5 arms and should not be reported
before deciding whether that condition is an acceptable benchmark outcome (it
is a Clean failure for every method, including Frozen). Do not select states
after seeing results.

## Next

Run the full Phase2A with the same script at 8 concurrency, including this
condition, and report it honestly. The pilot is not a main-table result.
