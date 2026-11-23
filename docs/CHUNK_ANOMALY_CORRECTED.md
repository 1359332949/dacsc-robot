# Corrected chunk-boundary analysis (retracts 89d6e51 conclusions)

## Retractions

1. `89d6e51` reported direction-reversal AUC .774/.943/1.000 as a failure signal.
   This was episode-length leakage: failures ran to the 220-step cap while
   successes ended earlier (73-181), so longer episodes accumulated more
   reversals. Fixed-prefix K=1 and K=3 reversal AUC is exactly .500 in all
   three factors. There is no early reversal warning signal.
2. `mean_disp` / `mean_cmd` AUC near 0 was misread as "no signal". Direction:
   **lower = failure**. The correct orientation is reported below.

## Fixed early prefixes (no length leakage)

Per-chunk displacement = ||state8_start[:3] - state8_end[:3]||.
Mean command = 0.05 * ||sum(command XYZ)|| (controller meters).
AUC is `P(fail < success)`; paired is within the same base_task.

| K (steps) | Factor | mean_disp AUC | paired | mean_cmd AUC | paired |
|---:|---|---:|---:|---:|---:|
| 1 (12) | Sensor | .710 | 19/23 (.826) | .665 | 19/23 |
| 1 (12) | Background | .824 | 19/21 (.905) | .814 | 19/21 |
| 1 (12) | Lighting | .312 | 4/6 | .243 | 4/6 |
| 3 (36) | Sensor | .792 | **22/23 (.957)** | .738 | 20/23 |
| 3 (36) | Background | .842 | **19/21 (.905)** | .824 | 16/21 |
| 3 (36) | Lighting | .778 | 4/6 | .875 | **6/6** |
| 5 (60) | Sensor | .751 | 18/23 | .765 | 19/23 |
| 5 (60) | Background | .824 | 20/21 (.952) | .842 | 20/21 |
| 5 (60) | Lighting | .847 | 4/6 | .708 | 4/6 |

Direction reversals in K=1/3: AUC .500 in all factors. K=5 reversal: Sensor
.649, Background .480, Lighting .521. Not a reliable early warning signal.

## Clean-normalized early-warning gate (important caveat)

Each shift trajectory divided by the same-base-task Clean K=3 mean.
Clean itself is ratio 1.0 (one Clean per task), so a threshold "ratio < 1"
has Clean FPR 0/10 and flags:

| Factor | failure recall | shift-success false alarm |
|---|---:|---:|
| Sensor | 12/13 (92.3%) | 11/17 (64.7%) |
| Background | 12/13 (92.3%) | 17/17 (100%) |
| Lighting | 6/6 (100%) | 22/24 (91.7%) |

So the normalized early displacement is **not yet a failure predictor**:
it separates clean from shift almost perfectly, and flags most successful
shift episodes too. It is a domain-shift / hypokinesia detector, not a
failure-specific gate. A within-episode temporal reference is required before
deployment.

## Engineering fixes

- `consensus_eval.py` medoid distance now normalizes by physical channel
  scales (XYZ .05 m, rotation .5 rad, gripper 2 range) instead of the previous
  squared weighting that made gripper 400x XYZ. The old Medoid background
  -3 result is therefore superseded and must not be cited as evidence against
  geometric-median consensus.
