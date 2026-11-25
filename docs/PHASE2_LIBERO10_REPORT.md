# Phase 2A: libero_10 5-arm fair physical table

Status: **RUNNING; no final table yet. Do not quote partial numbers.**

## Protocol

- 400 conditions = Clean 100 + bilateral Sensor 100 + Lighting 100 + Background 100.
- 10 official `libero_10` base tasks, init_index 1..10.
- 5 paired arms: frozen, pixel_align, deep_block12, actmad_true, ours_v1.
- Shared Spatial calibration; EGL/FP32; max_steps=520; 8 workers.
- Language uses base-task BDDL only; background is 5 table + 5 scene per task.
- Assertion `len(conditions)==400` and 100 per group passes.

## Progress at 17:34

```text
46 / 400 condition files complete
8 workers alive
```

Early observations (not final):
- Several libero_10 Clean/Lighting/Sensor conditions are hard for Frozen too; all arms often max out at 520.
- No per-arm conclusion can be drawn until all 400 conditions × 5 arms finish.

## Corrected Phase 1.5 sanity (after ActMAD reshape fix)

| Arm | Clean |
|---|---:|
| Frozen | 10/10 |
| V1 Block-3 | 10/10 |
| Pixel-Align | 9/10 |
| Deep Block-12 | 10/10 |
| True ActMAD | 10/10 |

## Corrected Table 4

| Arm | Median ms | x Frozen | Backward | Peak MB |
|---|---:|---:|---:|---:|
| Frozen | 22.38 | 1.00x | 0 | 397.6 |
| V1 Block-3 | 23.15 | **1.03x** | 0 | 397.6 |
| Pixel-Align | 23.63 | 1.06x | 0 | 397.9 |
| Deep Block-12 | 23.07 | 1.03x | 0 | 397.8 |
| True ActMAD (4-frame) | 71.98 | **3.22x** | 1 | 1099.5 |
| Multi-Sample Consensus N=8 | 180.50 | 8.07x | 0 | 398.2 |

The earlier 9.21x ActMAD number included an accidental per-step state_dict
clone; it is retracted.

## Calibration granularity

91 replan frames from 10 Spatial Clean init0 trajectories. This matches the
12-step CoRP replan cadence and is accepted.

## Next

Let the 2,000-rollout run finish, then analyze from JSON only (no hand-copied
numbers): per-group success, rescued/harmed vs Frozen and vs ActMAD, exact
two-sided McNemar, task-cluster 95% CI, table/scene split, five sensor subtypes.
