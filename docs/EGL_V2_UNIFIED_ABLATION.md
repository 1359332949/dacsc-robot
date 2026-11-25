# EGL V2 unified shallow-layer ablation (Spatial Dev, 3 seeds)

Status: **V2 candidates do not beat the frozen V1 anchor. Keep V1 frozen.**

Protocol: 130 conditions (100 standard Dev + 30 bilateral Sensor) x 3 policy
seeds = 390 paired conditions; 5 arms = 1,950 closed-loop rollouts. EGL, FP32,
`env.noise=0`, H=E=12, no EMA/grad/domain label. Matched hard alpha was locked
before outcomes from the t=0 soft activations of all 130 conditions: raw mean
0.4396, rounded to **0.45**.

## Standard Dev (100 x 3 = 300)

| arm | success | vs Frozen saved/lost/net | vs V1 saved/lost/net |
|---|---:|---:|---:|
| frozen | 217/300 | - | - |
| v1_b3_hard075 | **232/300** | 30/15/+15 | - |
| v2_prog_b3_b6 | 224/300 | 31/24/+7 | 17/25/**-8** |
| v2_soft_b3 | 228/300 | 29/18/+11 | 16/20/**-4** |
| v2_hard_matched_b3 | 222/300 | 24/19/+5 | 12/22/**-10** |

## Combined 130 (390)

| arm | success | vs Frozen | vs V1 |
|---|---:|---:|---:|
| frozen | 232/390 | - | - |
| v1_b3_hard075 | **256/390** | +24 | - |
| v2_prog_b3_b6 | 243/390 | +11 | **-13** |
| v2_soft_b3 | 246/390 | +14 | **-10** |
| v2_hard_matched_b3 | 243/390 | +11 | **-13** |

## Clean (30) and groups

- Clean: Frozen 29/30, V1 27/30, prog 27/30, soft **29/30**, matched 28/30.
  Only `v2_soft_b3` preserves the Frozen Clean count.
- Bilateral Sensor: Frozen 15/90, V1 24/90, prog 19, soft 18, matched 21.
- Single-eye Sensor: Frozen 56/90, V1 70/90, prog **73/90**, soft 68, matched 64.
- Table texture: Frozen 24/45, V1 22, prog 18, soft 19, matched 20.
- Scene texture: Frozen 34/45, V1 35, prog 32, soft **39**, matched 33.
- Lighting: Frozen 74/90, V1 78, prog 74, soft 73, matched 77.

## Interpretation

1. V2 progressive B3+B6 is worse than V1 on every aggregate: standard -8,
   combined -13, and it also loses 2 Clean cases. Not promoted.
2. V2 soft B3 beats the equal-strength hard matched arm by +6 standard and +3
   combined, and preserves Clean 29/30. Continuity helps relative to the
   matched hard control, but not enough to beat V1 (standard -4, combined -10).
3. Matched hard is also below V1; the loss is not explained by a larger
   intervention magnitude alone.
4. Background is split: soft improves scene (+5 vs Frozen) but loses table
   (-5); no stable Background gain.

Decision: **No V2 candidate is promoted. V1 remains the frozen reference.**
Task-cluster bootstrap intervals were not computed for this report; the raw
paired counts already show all V2 variants below V1. No Adapter, no target-test.
