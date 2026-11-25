# EAGA shallow feasibility probe V1 (2026-11-25)

Status: **attention survival passes; geometry/action probe fails; do not run closed loop.**

## Scope

Cached `4e850b8` Spatial Dev t=0 observations (100) and cached physical
bowl-move frames (37; only 32 nonzero-offset pairs were usable in this cache).
No closed-loop rollouts, no Object, no Blind, no gradients, no weights changed.

## Experiment 1: frozen CoRP attention survival

Extracted native `CrossAttentionResampler` weights per view, averaged over
heads and 48 queries -> 256 patch weights (sum=1). Clean reference is the
same-task Clean t=0 frame.

| Group | n | N_eff mean | N_eff max | cosine to Clean | top-25% IoU |
|---|---:|---:|---:|---:|---:|
| Clean | 10 | 46.2 | 101.2 | - | - |
| Sensor single (cache) | 30 | 52.2 | 129.4 | 0.410 | 0.343 |
| Background all | 30 | 61.5 | 109.4 | 0.628 | 0.353 |
| Table texture | 15 | 56.2 | - | 0.564 | 0.339 |
| Scene texture | 15 | 66.8 | - | 0.691 | 0.367 |
| Lighting | 30 | 55.0 | 115.6 | 0.800 | 0.594 |

Hard gate: **PASS** by the stated thresholds (`N_eff max < 180`,
table IoU mean 0.339 > 0.25). Caveat: the cache is single-eye Sensor, not the
bilateral stress set; the attention map is not uniformly collapsed, but
cosine/IoU show substantial drift under blur/texture.

## Experiment 2: pure-forward action probe

Three arms on the same cached t=0 inputs, fixed z0, no execution:
V1 uniform dead-zone alpha=0.75; EAGA weighted moments; EAGA spatial
foreground/background modulation (fg .35, bg .75).

Clean and Shift mean **normalized action change vs frozen** (lower is not
automatically better; it is a drift diagnostic):

| Group | V1 xyz | weighted xyz | spatial xyz |
|---|---:|---:|---:|
| Clean | 0.305 | 0.307 | 0.205 |
| Sensor | 0.193 | 0.206 | 0.175 |
| Background | 0.142 | 0.138 | 0.126 |
| Lighting | 0.271 | 0.204 | 0.208 |

Geometry probe (32 nonzero-offset physical bowl moves):

| Arm | median cosine | median amplitude ratio | cos>0 |
|---|---:|---:|---:|
| Frozen | 0.054 | 2.174 | 17/32 |
| V1 | -0.148 | 2.430 | 14/32 |
| EAGA weighted | -0.147 | 2.354 | 13/32 |
| EAGA spatial | -0.140 | 2.249 | 14/32 |

The cached geometry set is **not** the reported 97/108-pair set; only 32
nonzero-offset pairs are present here. Under this available probe, neither V1
nor EAGA preserves the Frozen geometry direction (`cos` near zero/negative),
and EAGA does not improve it.

## Decision

- Attention survival: pass, so the route is not physically impossible.
- Action/geometry: fail. EAGA does not meet the pre-declared 97/97 geometry
  gate; weighted moments and foreground/background modulation do not improve
  the direction response.
- **Do not run the 5-arm closed loop. Move to route 1 (second policy backbone
  validation of the minimal V1 principle).**

Limitations: single-eye cached Sensor; 32 geometry pairs from a 37-frame cache,
not the full 97/108 set; no task-cluster intervals; no closed-loop evidence.
