# Experiment audit: first-chunk residual and oracle comparisons

Status: corrective audit, not a method result.

## Confirmed correct

- The frozen split JSON is now enforced by `--split-json/--role`.
- Source fit IDs are disjoint from source validation and target test.
- `sample_z` experiments use a cached per-episode `z0` schedule shared across methods.
- Target Clean/Shift `state8` is identical at the first observation (max diff 0)
  for the collected pairs.
- The first-chunk Clean-action oracle is a privileged diagnostic; it is not
  deployable.

## Problems found

### A. Residual closed-loop comparison is not paired

`first_chunk_residual_eval.py` calls `policy.predict()` directly and does not
use the cached `z0` schedule. The Frozen baseline was produced by a different
sampler/run path. The residual +1/10 and global-mean 4/10 are therefore not
paired comparisons and cannot be called method effects.

### B. Source-paired residual uses source validation

`source_first_chunk_pairs.json` was collected on `source_fit + source_validation`.
The residual mapper was therefore fitted/retrieved with the source validation
states, so source validation is no longer an independent validation set.

This makes the experiment **known-perturbation paired simulator calibration**,
not target-only TTA.

### C. Inference interfaces are mixed

- source pairs: `policy.predict()`;
- target residual eval: `policy.predict()`;
- first-chunk oracle and Frozen `sample_z` baseline: `sample_z()` with `z0`;
- action diagnostic: `policy.predict()`.

Cross-interface numbers (oracle vs residual, visual oracle vs action oracle)
must not be treated as one paired experiment.

### D. Full physical pairing is not verified

Only 8-D state8 equality was checked. qpos/qvel, object state, camera pose and
intrinsics/extrinsics, and restore compatibility were not compared in the
residual/oracle script.

### E. Run stochasticity is visible

`frozen_predict` on target dev seed 2014 gave 4/10 in the action diagnostic and
5/10 in a separate rerun. This is consistent with unpaired flow sampling.

## Consequences

- Do not use 282104c kNN +1 or global-mean 4/10 as a method result.
- Do not use cross-interface oracle differences as a clean causal estimate.
- Reclassify 282104c as known-perturbation paired calibration.
- Keep the first-chunk oracle only as a privileged hypothesis, pending full
  pairing and shared-z0 verification.

## Required fixes before another closed-loop claim

1. Collect source and target pairs with one frozen sampler and a recorded
   per-state `z0` schedule.
2. Fit on source_fit only; keep source_validation untouched.
3. Run Frozen and residual with identical sampler, precision, chunking, and
   `z0` schedule.
4. Record full snapshot compatibility and camera config for paired runs.
5. Rerun the offline identifiability audit with paired actions; then rerun
   target-dev closed loop.

## Correction: kNN index direction and joint metric audit

The first version of `residual_identifiability_audit.py` computed the distance
matrix as `[N_source, N_target]` and then applied `argsort(axis=1)` as if it
were `[N_target, N_source]`. The kNN audit numbers in the first report were
therefore wrong and must not be used.

Corrected audit:

- distance matrix is `[N_target, N_source]`;
- both `state8`-only and joint `[state8, visual768]` metrics are reported;
- shuffled control is a 5000-permutation distribution, not one permutation.

Corrected direction results:

| predictor | cosine median | orthogonal error median |
|---|---:|---:|
| global source mean | 0.903 | 0.427 |
| state kNN k=1 | 0.803 | 0.591 |
| state kNN k=3 | 0.887 | 0.462 |
| state kNN k=5 | 0.874 | 0.485 |
| joint kNN k=1 | 0.741 | 0.671 |
| joint kNN k=3 | 0.855 | 0.516 |
| joint kNN k=5 | 0.894 | 0.447 |

Shuffled cosine median distribution: median 0.771, 5th pct 0.695, 95th pct
0.843.

Revised interpretation: joint k=1 is below the shuffled median and is not
evidence of local structure. Joint k=3/k=5 are above the shuffled 95th
percentile, so some local information is present, but neither exceeds the
global-mean direction. The first report's claim that "kNN adds noise" should
be narrowed to "joint k=1 is weak; k=3/5 have some local signal but do not
beat the global mean."

The closed-loop decision does not change: k=1 gave +1/10, k=3 gave 0, and
those rollouts were unpaired with the baseline. Do not restore the residual
route from this audit.
