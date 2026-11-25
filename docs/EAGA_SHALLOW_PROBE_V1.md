# EAGA shallow feasibility probe V1 (corrected v3, 2026-11-25)

Status: **attention survival valid; v2 action numbers retracted; v3 gives the
first valid 5-arm comparison. EAGA-spatial is a conditional candidate, not a
universal upgrade.**

Supersedes `d931d7f` and `d80611d` experiment-2 numbers. v2's source-moment
slicing, weighted-source cancellation, spatial reference binding and weighted
tau were all invalid. v3 rebuilds true t=0 source patches/attention and true
LOTO weighted references.

## Experiment 1: attention survival (unchanged, valid)

100 cached Spatial Dev t=0 cases; native resampler weights.

| Group | n | N_eff mean | N_eff max | cosine to Clean | top-25% IoU |
|---|---:|---:|---:|---:|---:|
| Clean | 10 | 46.2 | 101.2 | - | - |
| Sensor single (cache) | 30 | 52.2 | 129.4 | 0.410 | 0.343 |
| Background | 30 | 61.5 | 109.4 | 0.628 | 0.353 |
| Table texture | 15 | 56.2 | - | 0.564 | 0.339 |
| Scene texture | 15 | 66.8 | - | 0.691 | 0.367 |
| Lighting | 30 | 55.0 | 115.6 | 0.800 | 0.594 |

Gate: **PASS** (`N_eff max<180`, table IoU mean>0.25). Cached Sensor is
single-eye, not bilateral.

## Experiment 2: v3 five-arm comparison

All references fitted only on the 9 other tasks (LOTO 9 outer, 8+1 inner);
source t=0 patches and source attention are real. Four fixed noises, no
execution. `v1_full`/`spatial_full` use the full-trajectory uniform reference;
`v1_t0`/`spatial_t0` use the t=0 uniform LOTO reference; `weighted_t0` uses the
true weighted LOTO reference with its own weighted tau.

### Clean (10, 40 samples)

| Arm | active rate | xyz drift | rot drift |
|---|---:|---:|---:|
| v1_full | 0.0555 | 0.00182 | 0.00027 |
| spatial_full | 0.0555 | 0.00173 | 0.00027 |
| v1_t0 | 0.1289 | 0.00160 | 0.00027 |
| weighted_t0 | 0.1132 | 0.00724 | 0.00173 |
| spatial_t0 | 0.1289 | 0.00154 | 0.00027 |

### Relative action distance to same-task Clean Frozen (%; positive better)

| Group | v1_full | spatial_full | v1_t0 | weighted_t0 | spatial_t0 |
|---|---:|---:|---:|---:|---:|
| Sensor XYZ | 11.06 | 10.32 | **14.57** | 7.92 | 13.75 |
| Sensor rot | 10.02 | 10.56 | 13.23 | 3.24 | **13.47** |
| Background XYZ | 16.19 | **19.65** | 16.05 | 4.28 | **19.68** |
| Background rot | 14.51 | **15.61** | 13.79 | 8.20 | 15.35 |
| Lighting XYZ | 25.22 | 25.93 | **28.00** | 3.80 | 26.47 |
| Lighting rot | **18.66** | 16.82 | 18.97 | 12.54 | 17.23 |

Background split:

| Subgroup | v1_full | spatial_full | v1_t0 | weighted_t0 | spatial_t0 |
|---|---:|---:|---:|---:|---:|
| Table XYZ / rot | 16.58 / 9.82 | 19.38 / 12.36 | 17.21 / 12.23 | -2.22 / 3.46 | **19.97 / 14.57** |
| Scene XYZ / rot | 15.81 / 19.20 | **19.92 / 18.87** | 14.89 / 15.35 | 10.78 / 12.94 | 19.38 / 16.12 |

### Geometry (frozen-delta; 128 pair-noise comparisons)

| Arm | median cosine | median ratio |
|---|---:|---:|
| v1_full | 0.99979 | 0.99685 |
| spatial_full | 0.99983 | 0.99765 |
| v1_t0 | 0.99984 | 1.00772 |
| weighted_t0 | 0.99799 | 0.97214 |
| spatial_t0 | **0.99986** | **1.00690** |

All five pass the geometry-preservation gate; V1 values match the original
`4e850b8` behavior. This corrects the false "geometry failure" in v1/v2.

## Interpretation

1. `weighted_t0` is not competitive: it lowers Sensor/Background/Lighting
   relative gains and has the largest Clean drift. Weighted moments are rejected.
2. `spatial_full` is the only EAGA candidate with a real advantage: Background
   +3.46 XYZ / +1.10 rot over `v1_full`, table +2.80, scene +4.11; Clean drift is
   even slightly lower and geometry is preserved.
3. However it is **not a universal upgrade**: Sensor XYZ -0.74, Sensor rot
   +0.54, Lighting rot -1.84. The strict no-degradation gate is not met.
4. `v1_t0` suggests the t=0 LOTO reference itself is stronger than the full
   trajectory reference on Sensor/Lighting, but it is not an EAGA-specific gain.

## Decision

- Attention survival: **PASS**.
- Weighted EAGA: **REJECT**.
- Spatial EAGA: **conditional candidate only**. It passes geometry and Clean,
  and improves Background, but slightly degrades Sensor/Lighting rot. It is not
  promoted as a universal method.
- Do not run a large closed loop on the current evidence. If one small A/B is
  budgeted, run only `v1_full` vs `spatial_full` on Background + Sensor + Clean,
  3 seeds; otherwise stop route 2 and move to route 1.

Limitations: cached Sensor is single-eye; geometry cache has 32 nonzero pairs,
not 97/108; no closed-loop success; no task-cluster intervals.
