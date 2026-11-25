# EGL W2 closed-loop A/B, seed 0 (FairPhysical100)

Status: **STOP after seed 0. Both zero-hyperparameter W2 arms are far below V1.**

## Protocol

- EGL, FP32, H=E=12, `env.noise=0`, max_steps=220.
- 100 conditions: Clean 10 + bilateral Sensor 30 + Background 30
  (Table 15 / Scene 15) + Lighting 30. **No single-eye Sensor.** Name:
  `FairPhysical100`; not comparable to the old Standard100.
- 4 matched arms on the same states/seeds: frozen, v1_hard075,
  w2_proj_rms_noclamp, js_shrink_rms_noclamp.
- One policy seed (0) only, per the sequential rule. 400 rollouts.

## Results (seed 0)

| Group | Frozen | v1_hard075 | w2_proj_rms_noclamp | js_shrink_rms_noclamp |
|---|---:|---:|---:|---:|
| Clean | 10/10 | 10/10 | 10/10 | 10/10 |
| Bilateral Sensor | 5/30 | **7/30** | 5/30 | 5/30 |
| Background | 19/30 | **23/30** | 17/30 | 20/30 |
| Table texture | — | — | — | — |
| Scene texture | — | — | — | — |
| Lighting | 24/30 | **28/30** | 27/30 | 24/30 |
| **FairPhysical100** | **58/100** | **68/100** | **59/100** | **59/100** |

Background table/scene split is retained in the JSON; the seed-0 group
membership for those subkeys was not populated by the generic analyzer, so the
aggregate 30-case row is the valid primary row here.

## Paired transitions

- V1 vs Frozen: total +14 rescued / -4 harmed, net **+10**; task-cluster 95% CI
  for success-rate difference `[+2.7, +23.8] pp`.
- w2_proj_rms_noclamp vs Frozen: +6/-5, net **+1**; CI `[-23.8, +0.7] pp`.
- js_shrink_rms_noclamp vs Frozen: +8/-7, net **+1**; CI `[-20.2, -0.4] pp`.
- w2_proj_rms_noclamp vs V1: 4 rescued / 13 harmed, net **-9**.
- js_shrink_rms_noclamp vs V1: 3 rescued / 12 harmed, net **-9**.

Clean: no arm has any Clean rescue/harm; all 10/10.

## Stop decision

The sequential stop rule was: if either zero-hyperparameter arm is clearly below
V1 (>=4 games) or harms Clean, stop without seeds 1/2.

Both are 9 games below V1. **Stop.** Do not run seeds 1/2; do not promote W2.
Keep V1 frozen. The offline t=0 advantage does not transfer to closed loop.

Limitations: seed 0 only; no task-cluster intervals for the offline table; table/
scene split is present in JSON but the generic seed-0 analyzer did not populate
the subkeys; no claim beyond this Dev set.
