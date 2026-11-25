# Phase 1 / 1.5 report: libero_10 smoke, shared calibration, sanity, Table 4

## 1. libero_10 Clean smoke (Phase 1)

10 official base tasks, init_index=0, one deterministic env seed, EGL,
`env.noise=0`, max_steps=520, FP16 off (FP32), H=E=12.

**9/10 = 90%** successes; all actions finite. Failure: task 2
(KITCHEN_SCENE6 microwave mug), maxed at 520. Not tuned.
Details: `docs/data/phase1/libero10_clean_smoke.json`.

## 2. Shared source calibration (Phase 1.5)

Built from 10 `libero_spatial` Clean init_index=0 trajectories with Frozen.
Saved `runs/phase2_calib/spatial_shared_calib.npz` (copy in
`docs/data/phase15/spatial_shared_calib.npz`):

- pixel mean/std per view (RGB);
- per-block 0..11 patch mean/std;
- Block-3 and Block-12 LOTO 95% tau, shape (2, 384);
- 10 tasks, 91 replan frames / 182 view-frames total.

Limitation: the capture is per replan frame, not every control step; the
episodes terminate at task success/done. This is the full episode trajectory
available from that run, but it is coarse compared with the earlier full-source
files.

## 3. Clean sanity, init_index=1 (10 tasks)

| Arm | Success | Finite |
|---|---:|---:|
| Frozen | 10/10 | yes |
| V1 Block-3 | 10/10 | yes |
| Pixel-Align | **0/10** | yes |
| Deep Block-12 | 10/10 | yes |
| True ActMAD (all 12 layers, 1 SGD/replan) | 10/10 | yes |

Pixel-Align is catastrophic on Clean; it is not a viable baseline as
implemented. Deep Block-12 and ActMAD are identity-safe on Clean because their
dead-zone statistics are not strongly triggered on Clean; this is expected and
is not evidence of shift robustness.

## 4. Table 4: latency and memory

RTX 3090, batch 1, 50 Frozen warmups, 5 interleaved rounds x 30 measurements,
CUDA synchronize, no_grad for inference arms. ActMAD includes 1 update + forward.

| Arm | Median ms | IQR ms | x Frozen | Backward | Peak MB |
|---|---:|---:|---:|---:|---:|
| Frozen | 22.93 | 0.61 | 1.00x | 0 | 393.5 |
| Ours V1 Block-3 | 23.54 | 0.46 | **1.03x** | 0 | 393.5 |
| Pixel-Align | 24.03 | 1.86 | 1.05x | 0 | 393.8 |
| Deep Block-12 | 23.54 | 0.51 | 1.03x | 0 | 393.8 |
| True ActMAD (1-step SGD) | 72.09 | 3.77 | **3.14x** | 1 | 562.7 |
| Multi-Sample Consensus (N=8) | 183.56 | 7.60 | **8.00x** | 0 | 394.2 |

V1 is 1.03x Frozen, ~0.6 ms overhead, zero backward. ActMAD is 3.14x and
requires gradients; N=8 consensus is 8x. This is the clean deployment table.

## 5. Phase 1.5 decision

- libero_10 usable: 9/10 Clean.
- Shared calibration package exists and is frozen.
- True ActMAD implemented (all 12 layers, recent-8 buffer, 1 SGD/replan,
  episodic reset) and survives 10/10 Clean.
- Pixel-Align fails Clean; do not promote it.
- Table 4 is credible and V1's deployment advantage is now measured properly.
- Calibration granularity is the remaining caveat: 91 replan frames, not every
  control step.

Next: Phase 2 main table, using init_index>=2 for evaluation, shared calibration
from init0, bilateral Sensor only, official max_steps.
