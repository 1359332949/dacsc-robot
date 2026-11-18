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
