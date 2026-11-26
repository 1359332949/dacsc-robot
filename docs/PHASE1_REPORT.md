> **2026-11-26 对照命名更新：** 本文早期 ActMAD proxy/移植不能作为作者原版 ActMAD 的复现证据。请区分内部消融、明确移植与作者原版；相关实现审计见 `BASELINE_IMPLEMENTATION_AUDIT_V2.md`。历史数字不改写。

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

## 3. Corrected bugs before Phase 2

Audit found and fixed four implementation bugs:

1. `Calib.ref` and callers now use **0-indexed** block ids: `blocks[2] -> ref(2)`
   (Block 3), `blocks[11] -> ref(11)` (Block 12). Previously Block 3 was fed
   Block 4 statistics.
2. Pixel calibration and Pixel-Align now both use `[0,1]`; pixel counting is
   once per replan frame, not once per view. Assertions: pixel mean in
   `[0.05,0.95]`, std in `[0.05,0.50]`; Clean image mean abs diff `<30`.
3. `ActMAD.close()` restores the initial backbone weights and sets
   `requires_grad_(False)`, so it cannot contaminate later arms.
4. `ActMAD.update()` now stores the recent <=4 preprocessed frames and forwards
   them together with gradient; no detached history term. Parameter update L2
   is recorded.

## 4. Clean sanity, init_index=1 (10 tasks), corrected

| Arm | Success | Finite |
|---|---:|---:|
| Frozen | 10/10 | yes |
| V1 Block-3 | 10/10 | yes |
| Pixel-Align | **9/10** | yes |
| Deep Block-12 | 10/10 | yes |
| True ActMAD (12 layers, 1 SGD/replan, 4-frame window) | 10/10 | yes |

Pixel-Align is no longer a white screen: 9/10 Clean. Deep/ActMAD are identity-safe
on Clean; this is not shift evidence.

## 5. Corrected Table 4: latency and memory

RTX 3090, batch 1, 50 Frozen warmups, 5 interleaved rounds x 30 measurements,
CUDA synchronize; inference arms no_grad; ActMAD includes one update + forward.

| Arm | Median ms | IQR ms | x Frozen | Backward | Peak MB |
|---|---:|---:|---:|---:|---:|
| Frozen | 22.38 | 1.29 | 1.00x | 0 | 397.6 |
| Ours V1 Block-3 | 23.15 | 1.27 | **1.03x** | 0 | 397.6 |
| Pixel-Align | 23.63 | 1.05 | 1.06x | 0 | 397.9 |
| Deep Block-12 | 23.07 | 0.96 | 1.03x | 0 | 397.8 |
| True ActMAD (4-frame, 1-step SGD, no per-step clone) | 71.98 | 3.04 | **3.22x** | 1 | 1099.5 |
| Multi-Sample Consensus (N=8) | 180.50 | 1.46 | **8.07x** | 0 | 398.2 |


V1 is 1.03x Frozen, ~0.8 ms overhead, zero backward. True ActMAD is 3.22x
with one backward through the 4-frame window; the earlier 9.2x included an
accidental per-step full state_dict clone, now removed. N=8 consensus is 8.07x.

## 6. Phase 1.5 decision

- libero_10 usable: 9/10 Clean.
- Shared calibration package exists and is frozen.
- True ActMAD implemented (all 12 layers, 4-frame gradient window, 1 SGD/replan,
  episodic reset) and survives 10/10 Clean.
- Pixel-Align is fixed and now 9/10 Clean; keep as a baseline.
- Table 4 is credible and V1's deployment advantage is now measured properly.
- Calibration granularity is 91 replan frames, matching CoRP's 12-step replan
  cadence; this is the intended V1 granularity.

Next: Phase 2 main table, using init_index>=2 for evaluation, shared calibration
from init0, bilateral Sensor only, official max_steps.
