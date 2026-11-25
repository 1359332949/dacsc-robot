# EAGA shallow feasibility probe V1 (corrected 2026-11-25)

Status: **attention survival passes; corrected action/geometry probe fails; stop route 2.**

Supersedes the invalid experiment-2 numbers in `d80611d`. The four bugs in
`scripts/eaga_action_probe.py` were: double preprocessing, unused attention
weights, wrong Clean-relative reference, and wrong frozen-delta geometry
comparison. The corrected script is `scripts/eaga_action_probe_v2.py`.

## Experiment 1: attention survival (unchanged, valid)

Cached `4e850b8` Spatial Dev t=0, 100 cases. Native `CrossAttentionResampler`
weights per view, averaged over heads and 48 queries -> 256 patch weights.

| Group | n | N_eff mean | N_eff max | cosine to Clean | top-25% IoU |
|---|---:|---:|---:|---:|---:|
| Clean | 10 | 46.2 | 101.2 | - | - |
| Sensor single (cache) | 30 | 52.2 | 129.4 | 0.410 | 0.343 |
| Background all | 30 | 61.5 | 109.4 | 0.628 | 0.353 |
| Table texture | 15 | 56.2 | - | 0.564 | 0.339 |
| Scene texture | 15 | 66.8 | - | 0.691 | 0.367 |
| Lighting | 30 | 55.0 | 115.6 | 0.800 | 0.594 |

Gate: **PASS** by `N_eff max < 180` and table IoU mean > 0.25.
Caveat: cached Sensor is single-eye, not bilateral.

## Experiment 2: corrected action probe

Single preprocessing (`visual_from_obs` prepares exactly once), true weighted
target moments, Clean-relative action distance, and Frozen-delta geometry.
4 fixed policy noises; no execution; no weights changed.

### Identity check

V1 Clean absolute drift:

```text
xyz 0.00183, rot 0.00027
```

This is the expected small-drift regime (the buggy version reported 0.305).

### Relative action distance to same-task Clean Frozen (%)

Positive = closer to Clean than Frozen; negative = farther.

| Group | V1 xyz / rot | weighted xyz / rot | spatial xyz / rot |
|---|---:|---:|---:|
| Sensor | **+11.06 / +10.02** | -19.16 / -43.87 | +2.95 / -22.70 |
| Background | **+16.19 / +14.51** | -45.73 / -25.51 | -18.41 / -16.06 |
| Lighting | **+25.22 / +18.66** | -147.19 / -78.50 | -55.88 / -43.02 |

Clean absolute drift:

| Arm | xyz drift | rot drift |
|---|---:|---:|
| V1 | **0.00183** | **0.00027** |
| weighted | 0.05234 | 0.00998 |
| spatial | 0.03322 | 0.00634 |

### Geometry fidelity, corrected frozen-delta definition

`delta_a(adapt) = a_adapt(moved) - a_adapt(zero)`;
`delta_a(frozen) = a_frozen(moved) - a_frozen(zero)`.
Median cosine / amplitude ratio over 32 nonzero-offset pairs x 4 noises = 128.

| Arm | median cosine | median amplitude ratio |
|---|---:|---:|
| V1 | **0.99979** | **0.99685** |
| EAGA weighted | 0.82955 | 1.04584 |
| EAGA spatial | 0.96631 | 1.09327 |

## Decision

- Attention survival: **PASS**.
- Corrected action probe: V1 is positive on every domain and preserves the
  Frozen geometry response (cos 0.9998; ratio 0.9969).
- EAGA weighted moments and foreground/background modulation are **worse** on
  every Shift group, have 10-50x larger Clean drift, and degrade geometry.
- **Do not run closed loop; stop route 2 and move to route 1 (second policy
  backbone validation).**

Limitations: cached Sensor is single-eye; geometry cache has 32 nonzero pairs,
not the full 97/108 set; no closed-loop success; no task-cluster intervals.
