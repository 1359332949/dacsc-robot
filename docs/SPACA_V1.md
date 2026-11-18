# SPACA v1: Source-Paired Action-Consistent Adaptation

## Claim boundary

SPACA v1 is **paired-domain supervised behavioral calibration**, not
target-only unsupervised TTA. It uses source clean/shift pairs from the same
physical state, the same explicit flow-matching z0, and a frozen CoRP.

## Components

- frozen DINOv2 / Resampler / language / state / DiT;
- zero-initialized residual adapter on the 96x768 visual tokens;
- paired normalized-action loss (shift -> clean);
- clean action-preservation loss;
- no teacher, no feature residual loss, no smoothness loss in v1.

## Pair integrity

Pair collection uses full MuJoCo `get_state/set_state`, controller state,
episode flags, OSMesa, and explicit z0. Source-fit states 0/1 are train;
source-fit state 2 is held out. Target dev is evaluation only; target test is
never used.

## Offline held-out action error (source state 2)

| mode | RMSE | XYZ | Gripper |
|---|---:|---:|---:|
| frozen | 0.0687-0.0767 | 0.1017-0.1153 | 0.00151-0.00161 |
| adapter | 0.0444-0.0620 | 0.0636-0.0857 | 0.00178-0.00195 |

Held-out state 2 improves in overall/XYZ error but not gripper RMSE.

## Closed-loop target dev (OSMesa, same z0)

| state/domain/seed | frozen | adapter |
|---|---:|---:|
| 36 shift 7777 | 1/1 | 1/1 |
| 36 clean 7777 | 1/1 | 1/1 |
| 43 shift 7777 | 0/1 | **1/1** |
| 43 shift 8888 | 0/1 | **1/1** |
| 43 shift 9999 | 0/1 | **1/1** |
| 45 shift 7777 | 0/1 | 0/1 |

This is a pilot: 2 training states, 6 pair records, one adapter seed, and only
target-dev evaluation. Do not claim general SR improvement or target-test
generalization.

## Expanded source pairs (v1b)

Source fit states 0,1,2,4,6,8; two chunks each; 12 records, 11 usable.
Train states 0,1,2,4,6; validation state 8.

Offline held-out (state 8) summary:

- frozen RMSE 0.0721, XYZ 0.0964, gripper 0.00204;
- adapter RMSE 0.0223, XYZ 0.0227, gripper 0.00251.

The adapter improves overall/XYZ but **hurts gripper RMSE** on held-out state 8.
Closed-loop state 43 shift: frozen 0/1, adapter 1/1 (seed 7777).
State 45 shift: frozen 0/1, adapter 0/1.

Do not claim a general SR improvement. The gripper regression and state-45
failure are the main remaining defects.

## Gripper-weighted variant

Training uses per-dimension Huber weights with gripper weight 20.

Held-out state 8:

- frozen RMSE 0.0721, XYZ 0.0964, gripper 0.00204;
- adapter RMSE 0.0227, XYZ 0.0243, gripper 0.00143.

Gripper RMSE improves without losing overall/XYZ. Closed-loop state 43 still
0/1 -> 1/1; state 45 remains 0/1 -> 0/1.

This is the current best v1 adapter, but still target-dev pilot only.

## Full target-dev adapter-only failure and gate status

Adapter-only closed loop on target-dev shift (seed 7777):

| state | frozen | adapter |
|---|---:|---:|
| 34 | 0 | 0 |
| 35 | 0 | 0 |
| 36 | 1 | **0** |
| 37 | 1 | **0** |
| 39 | 0 | 0 |
| 43 | 0 | **1** |
| 45 | 0 | 0 |

Adapter-only is therefore not deployable: it rescues state 43 but breaks
states 36 and 37.

A monitor-gated variant was implemented. On state 36 it correctly kept the
frozen path (success), but on state 43 the monitor gate did not activate
(gate_used=false), so state 43 remained failure. The monitor gate needs a
separate calibration/debug pass before any full target-dev run.

Do not start the full target-dev adapter evaluation until the gate fix is
validated.

## Gate experiments (negative)

Adapter-only rescues state 43 but breaks states 36/37. Gate diagnostics:

- monitor persistence 15: gate activates at step 54; state 43 still fails;
- monitor persistence 1: gate activates at step 26; state 43 still fails;
- state 36 monitor max prob is 0.243, so gate correctly stays off;
- adapter-first-chunk preserves state 36/37 (both success) but does not rescue
  state 43.

Therefore no current gate solves both success preservation and recovery.
Adapter-only is not deployable. Do not start full target-dev / 20-episode.

## Phase 0 strict pairing audit (completed)

The pair collector is now fail-closed:

- strict `model_compatible` with body/geometry/physics checks;
- full `compare()` of qpos/qvel/act/ctrl/body/EEF (derived body/EEF
  tolerance 1e-3);
- camera geometry `camera_compare(expect_equal=True, tol=1e-3)`;
- snapshot digests and model/state/camera pass flags in every record;
- no swallowed observable exception.

Final source-fit collection (states 0,1,2,4,6,8; 2 chunks each):

```text
12 records
11 pairing PASS
1 record excluded: state 8 chunk 1, state8_max=1.18e-3
```

The earlier 11 pairs are provisionally invalid for strict claims; they predate
the full model/state/camera audit.

Gate diagnostics:

- `gate_step` is the monitor trigger time, not the adapter application time;
- adapter actually applies at the next replan boundary;
- monitor persistence 1 triggers at step 26; state 43 still fails;
- first-chunk adapter preserves 36/37 but does not rescue 43.

No adapter retraining or full target-dev rollout was performed in Phase 0.

## Strict multi-z0 adapter (v1c)

Data: final strict pairs (11 pass; state 8 chunk 1 excluded).
Training: pair + identity + z0 augmentation (4 z0) + residual 0.01;
train states 0,1,2,4,6; validation state 8.

Closed-loop target-dev shift, seed 7777:

| state | frozen | adapter |
|---|---:|---:|
| 36 | 1 | 1 |
| 37 | 1 | 1 |
| 43 | 0 | 1 |
| 45 | 0 | 1 |

Clean target-dev:

| state | frozen | adapter |
|---|---:|---:|
| 36 | 1 | 1 |
| 37 | 1 | 1 |
| 43 | 1 | 1 |

But multi-seed is unstable:

- state43: adapter 1/1 at seed7777, 0/1 at seed8888, 0/1 at seed9999;
- state45: adapter 1/1 at seed8888; seed9999 pending/not robust.

Conclusion: multi-z0 training removed the 36/37 breakage in this seed and
gives a strong seed-7777 pilot, but the recovery is still seed-sensitive.
Do not claim general SR improvement or start full target-dev confirmation.
