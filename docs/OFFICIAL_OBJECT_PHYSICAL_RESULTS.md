# Official LIBERO-Object physical cross-suite table (completed 2026-11-25)

Status: **completed cross-suite benchmark re-test; not a blind/held-out claim.**
This is the pre-registered table from `OFFICIAL_OBJECT_CROSS_SUITE_PREREG.md`.

## Protocol

- Suite: official `libero_object`, 10 base tasks.
- 130 conditions: 10 independent Clean, 30 official Lighting, 30 official
  Background, 30 official single-eye Sensor, 30 bilateral Sensor stress
  (same official kernel applied to both views).
- 3 policy seeds; 3 arms; 1,170 closed-loop rollouts; EGL; FP32; H=E=12;
  `env.noise=0`; no Adapter, no gradient, no external visual model.
- Arms: `frozen`; `spatial_ref` (global V1 reference fitted only on the 10
  Spatial Clean trajectories, applied zero-shot); `object_clean_cal` (Object
  Clean LOTO calibration, privileged calibration arm, not zero-shot).
- Pairs are matched by (task, instance, seed). Task-cluster bootstrap 95% CI
  over the 10 tasks; 10 clusters only.

## Results

| Group | Frozen | spatial_ref | object_clean_cal |
|---|---:|---:|---:|
| Clean | 30/30 | **30/30** | 29/30 |
| Sensor single | 53/90 | 71/90 | 70/90 |
| Sensor bilateral | 20/90 | 22/90 | 24/90 |
| Background | 41/90 | 49/90 | 50/90 |
| Lighting | 86/90 | 87/90 | 88/90 |
| Standard100 | 210/300 | **237/300** (+38/-11, net +27) | **237/300** (+38/-11, net +27) |
| Combined130 | 230/390 | **259/390** (+40/-11, net +29) | **261/390** (+43/-12, net +31) |

Task-cluster 95% CI for spatial_ref vs Frozen:
- Standard100: +9.0 pp, CI [5.0, 13.0] pp.
- Combined130: +7.4 pp, CI [4.1, 11.0] pp.

Object calibration vs spatial_ref:
- Standard100: 0.0 pp, CI [0.0, 0.0] (tie).
- Combined130: +0.5 pp, CI [0.0, 1.8] (tiny); Clean loses 1 case.

## Interpretation

1. The frozen V1 Spatial reference transfers to official Object physical
   Lighting/Background/Sensor without Object outcomes: standard +27/300,
   combined +29/390, Clean 30/30 with zero rescue/harm.
2. Object Clean calibration is not better than the Spatial zero-shot reference
   on the standard set (tie) and only +0.5 pp on combined, while losing one
   Clean case. It is not needed for the main method.
3. Bilateral Sensor gain is small (+2/90; CI [0, 6.7]) and cannot support a
   "solves bilateral blur" claim; single-eye Sensor is a benchmark artifact and
   must not be used as method evidence.
4. Background is the main driver (+8/90), Lighting is nearly saturated (86/90).

## Limitations

- Historical access to the official Object pool is not exhaustively audited;
  this is a benchmark re-test, not Blind.
- The Object Clean calibration reference is an existing frozen artifact; it is
  reported as privileged, not zero-shot.
- Only 10 task clusters; CIs are descriptive, not a new-task guarantee.
- No method was selected or tuned on these outcomes after the fact.

Evidence: `docs/data/official_object_physical_manifest.json`,
`docs/data/official_object_physical_summary.json`,
`scripts/official_object_cross_suite.py`.
