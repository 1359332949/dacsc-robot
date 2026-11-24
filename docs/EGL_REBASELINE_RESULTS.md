# EGL/GPU rendering rebaseline (2026-11-24)

Decision: switch the closed-loop benchmark to the standard `MUJOCO_GL=egl`
GPU renderer. No EGL method result is compared with an OSMesa baseline.

## Frozen baseline, 3 policy seeds

Same 100 Dev cases, same initial states, same language, same controller,
`max_steps=220`. Seed 0 uses the original per-case policy flow seed; seeds 1/2
use independent deterministic streams.

| Seed | Clean | Sensor | Background | Lighting | Total |
|---|---:|---:|---:|---:|---:|
| 0 | 10/10 | 20/30 | 19/30 | 24/30 | 73/100 |
| 1 | 9/10 | 19/30 | 20/30 | 24/30 | 72/100 |
| 2 | 10/10 | 17/30 | 19/30 | 26/30 | 72/100 |
| Mean | 9.67/10 | 18.67/30 | 19.33/30 | 24.67/30 | 72.33/100 |

This is the new EGL reference. It is materially stronger than the OSMesa
reference (Clean 10, Sensor 17, Background 17, Lighting 24), confirming that
mixing backends would have inflated method gains.

## Zero-training consensus under EGL, 3 seeds

z0=0 is deterministic in policy noise; its three repeats still vary because the
Sensor noise stream/observation boundary is replayed in separate processes.

| Method | Clean mean | Sensor mean | Background mean | Lighting mean | Total mean |
|---|---:|---:|---:|---:|---:|
| Frozen EGL | 9.67 | 18.67 | 19.33 | 24.67 | 72.33 |
| z0=0 | 9.00 | 18.00 | 20.33 | 24.00 | 71.33 |
| Trimmed-Mean | 9.67 | 18.33 | 19.33 | 23.33 | 70.67 |

Paired net transitions vs the same-seed Frozen baseline:
- z0=0: seed 0 -2, seed 1 -1, seed 2 0; mean -1.0.
- Trimmed: seed 0 -1, seed 1 -2, seed 2 -2; mean -1.67.

Neither operator has a positive three-seed net gain under EGL. The earlier
OSMesa conclusion that z0=0 improves Sensor and Background is not reproduced.
The Background +1 average for z0=0 is offset by Sensor/Lighting/Clean losses.

## Decision

No method is promoted. Keep the EGL Frozen baseline as the only reference.
Do not mix OSMesa and EGL numbers. Retire z0=0 / Trimmed / Medoid as main
methods unless a new pre-registered evaluation reverses this result.

Evidence: docs/data/egl_baseline_seed{0,1,2}.json and
docs/data/egl_consensus_s{0,1,2}.json. No Adapter, no target-test.
