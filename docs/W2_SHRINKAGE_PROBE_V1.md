# W2 shrinkage zero-hyperparameter probe v1 (2026-11-25)

Status: **completed; no zero-hyperparameter variant beats frozen V1.**

## Protocol

Cached `4e850b8` Spatial Dev: 100 t=0 inputs and 32 nonzero-offset bowl-move
pairs, 4 fixed policy noises. No environment rollout, no gradients, no weights
changed. Source statistics from `full.npz`; source natural RMS radius computed
from `runs/full_traj_v1/source/*.npy` with correct moment `unbind(1)`, outer-9 /
inner-8+1 LOTO. All arms use the same W2 geodesic update and clamp [0.5, 2.0].

`r_rms` per task is 0.1359–0.1401.

## Clean

| Arm | mean alpha | active rate | XYZ drift | Rot drift |
|---|---:|---:|---:|---:|
| v1_hard075 | 0.042 | 0.055 | 0.00182 | 0.00027 |
| w2_proj_q95 | 0.007 | 0.055 | **0.00031** | **0.00010** |
| js_shrink_q95 | 0.013 | 0.055 | 0.00062 | 0.00012 |
| w2_proj_rms | 0.065 | 0.279 | 0.00250 | 0.00042 |
| js_shrink_rms | 0.107 | 0.279 | 0.00360 | 0.00047 |

## Relative action distance to same-task Clean Frozen (%; positive better)

| Group | v1_hard075 | w2_proj_q95 | js_shrink_q95 | w2_proj_rms | js_shrink_rms |
|---|---:|---:|---:|---:|---:|
| Sensor XYZ | **11.06** | 7.57 | 8.53 | 13.80 | 16.26 |
| Sensor Rot | **10.02** | 6.46 | 7.97 | 8.99 | 9.98 |
| Background XYZ | **16.19** | 15.98 | 16.79 | 16.79 | 17.34 |
| Background Rot | **14.51** | 14.04 | 15.21 | 13.35 | 13.84 |
| Lighting XYZ | 25.22 | 23.05 | 22.40 | **26.76** | 26.29 |
| Lighting Rot | 18.66 | 17.10 | 19.15 | **22.49** | 23.03 |

Background split:

| Subgroup | v1 | w2_q95 | js_q95 | w2_rms | js_rms |
|---|---:|---:|---:|---:|---:|
| Table XYZ / rot | **16.58 / 9.82** | 13.11 / 8.57 | 15.19 / 11.76 | 16.35 / 11.61 | 17.70 / 14.59 |
| Scene XYZ / rot | 15.81 / 19.20 | **18.85 / 19.51** | 18.39 / 18.66 | 17.22 / 15.08 | 16.98 / 13.08 |

## Geometry (frozen-delta; 128 pair-noise comparisons)

| Arm | median cosine | median ratio |
|---|---:|---:|
| v1_hard075 | 0.99979 | 0.99685 |
| w2_proj_q95 | **0.99999** | **1.00130** |
| js_shrink_q95 | 0.99997 | 1.00154 |
| w2_proj_rms | 0.99985 | 0.99977 |
| js_shrink_rms | 0.99965 | 1.00012 |

Clamp trigger fraction is below 1.6% in every arm, so clamp is not doing the work.

## Interpretation

1. The W2/James–Stein closed forms are real and geometry-safe, and the Clean
   no-op is cleaner than V1. But they do **not** beat V1 on the Shift groups.
2. `q95` variants under-correct: Sensor 7.6/6.5 vs V1 11.1/10.0; Background
   slightly worse; Lighting worse.
3. `rms` variants over-correct: Sensor XYZ 13.8–16.3, but Clean drift 1.4–2.0x
   V1 and Background/Lighting mixed.
4. V1's fixed alpha=.75 is not arbitrary in this probe: it sits between the
   under-correcting q95 projection and the over-correcting RMS projection.
   Removing alpha does not improve the overall result.
5. No variant is promoted. V1 remains the frozen method; route 2/W2 closed.

Limitations: single-eye cached Sensor; 32 nonzero geometry pairs; t=0 only;
no closed-loop success; no task-cluster intervals.
