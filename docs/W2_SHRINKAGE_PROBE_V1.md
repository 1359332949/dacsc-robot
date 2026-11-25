# W2 shrinkage probe (corrected v2, 2026-11-25)

Status: **v1 scalar-r bug retracted; v2 per-channel RMS shows a competitive
zero-hyperparameter candidate, but not a clean sweep.**

## What changed

v1 line 40 collapsed the per-view/per-channel natural radius into one scalar
(0.139). That inflated RMS activation and invalidated the RMS arms. v2 rebuilds
`r_rms` as `(2, 384)` with frame-level LOTO fitting:

- inner 8 tasks: concatenate all frames in float64, fit mean/second moment;
- score every held task frame;
- concatenate 9 held-fold scores, mean over frames, then sqrt per channel.

Assertion: `r_rms.shape==(2,384)`, finite, `max/min>1.5`. Passed for all 10 tasks.

## Clean

| Arm | mean alpha | active rate | XYZ drift | Rot drift |
|---|---:|---:|---:|---:|
| v1_hard075 | 0.042 | 0.055 | 0.00182 | 0.00027 |
| w2_proj_q95 | 0.007 | 0.055 | **0.00031** | **0.00010** |
| js_shrink_q95 | 0.013 | 0.055 | 0.00062 | 0.00012 |
| w2_proj_rms | 0.084 | 0.347 | 0.00172 | 0.00024 |
| js_shrink_rms | 0.139 | 0.347 | 0.00277 | 0.00038 |
| js_shrink_rms_noclamp | 0.139 | 0.347 | 0.00277 | 0.00038 |

## Relative action distance to same-task Clean Frozen (%; positive better)

| Group | v1 | w2_q95 | js_q95 | w2_rms | js_rms | js_rms_noclamp |
|---|---:|---:|---:|---:|---:|---:|
| Sensor XYZ | 11.06 | 7.57 | 8.53 | 11.22 | **13.74** | 13.05 |
| Sensor Rot | **10.02** | 6.46 | 7.97 | 8.47 | 9.30 | 9.45 |
| Background XYZ | 16.19 | 15.98 | 16.79 | 18.09 | **17.98** | **19.65** |
| Background Rot | 14.51 | 14.04 | 15.21 | 15.99 | 16.06 | **17.51** |
| Lighting XYZ | **25.22** | 23.05 | 22.40 | 24.97 | 23.04 | 23.00 |
| Lighting Rot | 18.66 | 17.10 | 19.15 | **20.21** | 20.82 | **21.20** |

Background split:

| Subgroup | v1 | w2_q95 | js_q95 | w2_rms | js_rms | js_rms_noclamp |
|---|---:|---:|---:|---:|---:|---:|
| Table XYZ / rot | 16.58 / 9.82 | 13.11 / 8.57 | 15.19 / 11.76 | 17.08 / 13.02 | **18.29 / 16.34** | **20.67 / 18.16** |
| Scene XYZ / rot | 15.81 / 19.20 | **18.85 / 19.51** | 18.39 / 18.66 | 19.11 / 18.97 | 17.67 / 15.78 | 18.62 / 16.85 |

## Geometry (frozen-delta; 128 pair-noise comparisons)

| Arm | median cosine | median ratio |
|---|---:|---:|
| v1_hard075 | 0.99979 | 0.99685 |
| w2_proj_q95 | **0.99999** | **1.00130** |
| js_shrink_q95 | 0.99997 | 1.00154 |
| w2_proj_rms | 0.99975 | 0.99379 |
| js_shrink_rms | 0.99928 | 0.99077 |
| js_shrink_rms_noclamp | 0.99928 | 0.99077 |

Clamp trigger fraction is below 1.7%; `noclamp` changes almost nothing.

## Interpretation

1. Correct per-channel `r_rms` removes the v1 Clean activation artifact but does
   not produce a universal win. Clean active rate is 0.347 and Clean drift is
   ~1.5x V1, not zero.
2. `js_shrink_rms` wins Sensor XYZ (+2.68), Background XYZ (+1.79), Table
   (+1.71 / +6.52) and Lighting rot (+2.16), but loses Sensor rot (-0.72),
   Scene XYZ/rot, Lighting XYZ, and Clean drift. It is a **competitive
   candidate**, not a passing universal upgrade.
3. `js_shrink_rms_noclamp` is essentially identical to the clamped version;
   clamp is not carrying the result. Dropping clamp is safe for theory but does
   not improve the metrics.
4. `q95` variants under-correct. Fixed-V1 remains the safest point estimate.

## Decision

- v1 W2 numbers are retracted.
- `js_shrink_rms` is the only zero-hyperparameter arm worth a controlled
  closed-loop A/B against V1; it is not promoted on t=0 evidence alone.
- Recommended minimal A/B: V1 vs `js_shrink_rms_noclamp`, 3 seeds, Clean +
  Sensor bilateral + Background table + Lighting; report rescue/harm and
  geometry. Do not run a large tournament.
- V1 remains frozen until that A/B finishes.

Limitations: cached Sensor is single-eye; 32 nonzero geometry pairs; t=0 only;
no closed-loop success; no task-cluster intervals.
