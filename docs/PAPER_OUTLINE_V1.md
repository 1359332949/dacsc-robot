# Paper outline V1 (2026-11-25)

Working title: **A Frozen-Policy Shallow-Layer Distribution Adapter for
Visually Shifted Robot Manipulation: What Works, What Does Not, and How to
Prove It**

## Abstract (draft)

We study test-time visual robustness of a frozen compact manipulation policy
(CoRP) under bilateral visual corruption. We show that a single shallow,
source-only distribution alignment at DINOv2 Block 3 (hard 95% dead-zone,
alpha=0.75, full-trajectory leave-one-task-out reference) improves EGL Spatial
Dev from 217/300 to 232/300 over three policy seeds while preserving physical
geometry (97/97 four-layer bowl-move checks) and not increasing aggregate
failure. We then report a deliberately broad negative ablation: two-layer
progressive Block 3+6 alignment, continuous soft gating, equal-strength hard
gating, Medoid/Trimmed/z0 Flow consensus, local patch energy, channel
suppression, and t=0 handcrafted risk signals all fail to beat the frozen V1
anchor. We argue that most reported gains in this setting are either
backend-specific (OSMesa vs EGL), single-eye benchmark artifacts, or
offline-distance illusions. The paper contributes a protocol-first negative
map plus one frozen, deployable operator, and an explicit boundary between
development reuse and blind confirmation.

## Contributions

1. A frozen, zero-training shallow-layer operator: Block-3 source-only
   full-trajectory LOTO distribution alignment, hard dead-zone, alpha=0.75.
2. A rigorous EGL paired protocol: contemporaneous Frozen, 3 policy seeds,
   bilateral Sensor stress, rescue/harm counts, no cross-renderer mixing.
3. A negative atlas: multi-layer progressive alignment, soft gating,
   equal-strength hard control, Flow consensus, channel/spatial suppression
   and handcrafted risk signals do not beat V1.
4. A leakage/benchmark audit: single-eye Sensor is a construct artifact and is
   excluded from method evidence; official-pool historical access is disclosed.
5. A cross-suite frozen transfer screen on Object synthetic and a pre-registered
   official Object physical table (not yet run).

## Method

- Representation: frozen CoRP, DINOv2 ViT-S/14, 2 views x 48 tokens; no external
  model, no VLM, no gradient, no EMA, H=E=12.
- Reference: 10 Spatial Clean seed-0 trajectories, 101 inference chunks;
  source-only LOTO (outer 9 tasks, inner 8+1), per-channel mean/std/tau95 at
  Block 3 (`blocks[2]`).
- Operator: for channels with normalized distance > tau, bounded hard
  dead-zone alignment with alpha=0.75; both views symmetric; inactive channels
  exactly bypassed.
- No domain label, no frame-length feature, no test-time label.

## Experiments

### A. Mechanism and safety

- Four-layer physical bowl-move geometry: 97/97 preserved.
- Full-trajectory reference: Clean 0/0 rescue/harm; geometry unchanged.
- EGL/OSMesa audit: EGL baseline is materially stronger; no mixing.

### B. Spatial Dev (EGL, 3 seeds, 100 conditions = 300)

| Arm | Clean | Sensor | Background | Lighting | Standard |
|---|---:|---:|---:|---:|---:|
| Frozen | 29/30 | 56/90 | 58/90 | 74/90 | 217/300 |
| V1 Block3 hard075 | 27/30 | 70/90 | 57/90 | 78/90 | **232/300** |
| V2 prog B3+B6 | 27/30 | 73/90 | 50/90 | 74/90 | 224/300 |
| V2 soft B3 | 29/30 | 68/90 | 58/90 | 73/90 | 228/300 |
| V2 matched hard | 28/30 | 64/90 | 53/90 | 77/90 | 222/300 |

Note: single-eye Sensor is shown only as a benchmark-construction diagnostic;
the fair comparison is bilateral Sensor 90 (next table).

### C. Fair bilateral Sensor (30 conditions x 3 seeds = 90)

| Arm | Bilateral Sensor |
|---|---:|
| Frozen | 15/90 |
| V1 | **24/90** |
| V2 prog | 19/90 |
| V2 soft | 18/90 |
| V2 matched | 21/90 |

Decision: V2 not promoted; V1 frozen.

### D. Flow consensus negative result (OSMesa legacy, not paper main table)

Frozen 58/90; z0=0 60/90; Trimmed 60/90; Medoid 53/90. Under the EGL
rebaseline these gains do not reproduce: EGL Frozen 72.33/100, z0=0 71.33,
Trimmed 70.67. Report as backend-specific negative evidence, not main method.

### E. Object synthetic cross-suite screen (frozen candidate)

Standard100 Frozen 287/300, Spatial reference 293, Object LOTO calibration 295.
Bilateral Sensor 40/90 -> 52/90 -> 54/90; Clean 30/30. This is a new synthetic
perturbation screen, not official LIBERO-Plus holdout; Object clean/smoke rows
are excluded from the candidate jobs and the role is disclosed.

### F. Pre-registered official Object physical table (next)

See OFFICIAL_OBJECT_CROSS_SUITE_PREREG.md. Not run yet. It is a cross-suite
benchmark re-test, not a blind claim.

## Limitations

- 10 base tasks; repeated instances and seeds are not independent tasks.
- Spatial Dev is historical reuse; no zero-contact Blind exists.
- Object synthetic perturbations are generated, not official assets.
- V1 clean count in the V2 batch was affected by a per-arm EGL rounding cache
  bug (fixed in code; full V2 rerun not repeated because the paired ranking is
  unchanged). The earlier V1 report is 233/300; this batch is 232/300.
- No task-cluster bootstrap intervals for the V2 table yet.
- Actual intervention magnitude |gamma-1| not logged in the V2 batch.

## Next actions

1. Freeze the V1 method and the paper claims above.
2. Run official Object physical Lighting/Background/Sensor + bilateral Sensor
   + independent Clean, three arms (Frozen / Spatial reference / Object Clean
   calibration), 3 seeds, EGL, with task-cluster intervals.
3. Only after that, write the final main table and figures. Do not open a new
   Spatial V3/V4 branch.
