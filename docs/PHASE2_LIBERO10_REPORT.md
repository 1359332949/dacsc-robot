> **2026-11-26 对照实现审计更新：** 本文旧 `actmad_true` 是自行移植版本，不能当作原版 ActMAD 的忠实复现；其多帧双相机排列存在已确认错误。修复后实现标识为 `actmad_inspired_viewwise_v2`，已通过真实模型执行链检查，但尚未重新取得闭环对比成绩。原数字保留作历史记录，不能据此主张优于原版 ActMAD。后续共享校准阈值另有公式问题，须分别追溯。见 `BASELINE_IMPLEMENTATION_AUDIT_V2.md`。

# Phase 2A complete: libero_10 5-arm fair physical table

## Protocol

- 400 conditions = Clean 100 + bilateral Sensor 100 + Lighting 100 + Background 100.
- 10 official libero_10 base tasks, init_index 1..10.
- 5 paired arms: Frozen, Pixel-Align, Deep Block-12, true ActMAD, Ours V1.
- Shared Spatial init0 calibration; EGL/FP32; max_steps=520; 8 workers.
- Language uses base-task BDDL only. Background = 5 table + 5 scene per task.
- 2,000 rollouts total; all 400 condition files complete.

## Success

| Group | n | Frozen | Pixel | Deep B12 | ActMAD | Ours V1 |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 100 | 95/100 | 74/100 | 91/100 | 95/100 | 90/100 |
| Sensor bilateral | 100 | 14/100 | 2/100 | 14/100 | 14/100 | **17/100** |
| – Motion blur | 20 | 12/20 | 2/20 | 14/20 | 13/20 | **15/20** |
| – Gaussian blur | 20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| – Zoom blur | 20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| – Fog | 20 | 0/20 | 0/20 | 0/20 | 0/20 | 0/20 |
| – Glass blur | 20 | 2/20 | 0/20 | 0/20 | 1/20 | 2/20 |
| Lighting | 100 | 53/100 | 28/100 | 57/100 | 50/100 | **62/100** |
| Background | 100 | 28/100 | 14/100 | 28/100 | 28/100 | 28/100 |
| – Table | 50 | 15/50 | 8/50 | 14/50 | 15/50 | **16/50** |
| – Scene | 50 | 13/50 | 6/50 | **14/50** | 13/50 | 12/50 |
| **Total** | **400** | **190** | **118** | **190** | **187** | **197** |

## Paired transitions vs Frozen

| Group | Pixel | Deep B12 | ActMAD | Ours V1 |
|---|---:|---:|---:|---:|
| Clean | **-21** (p=1.1e-5) | -4 (p=.78) | 0 (p=1) | **-5** (p=.45) |
| Sensor | **-12** (p=.001) | 0 | 0 | +3 (p=1) |
| Lighting | **-25** (p=4.5e-5) | +4 (p=.85) | -3 (p=1) | **+9** (p=.16) |
| Background | **-14** (p=.019) | 0 | 0 | 0 |
| **Total** | **-72** (p=3.7e-14) | 0 (p=1) | -3 (p=1) | **+7** (p=.87) |

## Paired transitions vs ActMAD

| Group | Frozen | Pixel | Deep B12 | Ours V1 |
|---|---:|---:|---:|---:|
| Clean | 0 | -21 (p=.0002) | -4 | -5 |
| Sensor | 0 | -12 (p=.001) | 0 | +3 |
| Lighting | +3 | -22 (p=.0009) | +7 (p=.19) | **+12** (p=.008) |
| Background | 0 | -14 (p=.025) | 0 | 0 |
| **Total** | +3 | -69 (p=1.7e-11) | +3 | **+10** (p=.41) |

## What this says

1. **V1 is the best total arm** (197/400 vs Frozen 190), but the +7 is not
   statistically significant and is not a Clean-safe win: V1 loses 5 Clean
   successes. It gains +9 on Lighting and +3 on Sensor, and is flat on
   Background.
2. **The Clean regression is the main red flag.** Any method that loses Clean
   while claiming a TTA win needs a gate or a stronger no-op; V1 does not meet
   the pre-declared Clean guardrail on libero_10.
3. **Pixel-Align is catastrophic** (-72 total, p=3.7e-14); it should stay as a
   weak baseline, not a serious competitor.
4. **Deep Block-12 and true ActMAD are identity on this suite** (190/187), with
   no significant gains; ActMAD costs 3.22x latency.
5. **Sensor is floor-bound**: Gaussian/Zoom/Fog are 0/20 for every method; only
   Motion blur carries signal, and V1 leads there (15/20 vs Frozen 12/20).
6. **Background has no method signal**; V1 is literally 28/100, same as Frozen.

## Decision

- V1 remains the best total arm but **fails the Clean guardrail** on libero_10.
- Do not claim a universal gain on this suite.
- Next: either add a Clean-preserving gate for V1 or report libero_10 honestly
  as a mixed/negative transfer suite. Do not tune on this test set.

## Task-cluster 95% CI vs Frozen

| Group | Pixel | Deep B12 | ActMAD | Ours V1 |
|---|---:|---:|---:|---:|
| Clean | [-32,-9] | [-12,+3] | [-4,+5] | **[-9,-1]** |
| Sensor | [-16,-8] | [-6,+6] | [-6,+5] | [-6,+10] |
| Lighting | [-45,-5] | [-3,+12] | [-11,+4] | [0,+20] |
| Background | [-22,-6] | [-7,+7] | [-8,+8] | [-8,+7] |
| Total | [-25,-11] | [-4.5,+3.5] | [-3.3,+1.3] | [-3.3,+7.0] |

V1's Clean interval excludes zero: **the Clean loss is a real regression on
libero_10**, even though total is nominally +7. The total interval includes zero.

