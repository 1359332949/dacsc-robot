# Clean-retention and rotation-risk repair

## Re-audit of v2 Clean metrics

The v2 result has 12 decision-state records (4 trajectories x 3 chunks).
The quoted values 0.1095 / 0.0923 / 0.0261 match the first record (state 3,
chunk 0), not the mean over all 12 records.

Re-aggregating every record gives these full-chunk normalized action costs:

| Clean-side diagnostic | Mean over 12 records |
|---|---:|
| Shift-derived raw Target-only correction applied to Clean twin | 0.14213 |
| Shift-derived raw content-filter correction applied to Clean twin | 0.12986 |
| Clean-stream raw Target-only estimator | 0.12306 |
| Clean-stream content-filter estimator | 0.09636 |

The first two are cross-domain transfer stress tests, not operational Clean-only
retention. The latter two are the relevant Clean-stream negative controls and
are still not close to identity. The new validator reports all four separately,
with full/prefix metrics and per-episode aggregates.

## Source-calibrated abstention

The previous estimator always applied an EMA correction, including when the
residual resembled source-Clean prediction noise or the current proprioceptive
context was outside the source calibration support.

The new source-only confidence gate combines:
- residual novelty beyond the cross-fitted source-Clean residual quantile;
- nearest-neighbour support in the source proprioceptive-context set.

Confidence is zero for source-noise residuals and for contexts beyond twice the
leave-one-out source support radius. It decays continuously between thresholds.
This is an abstention heuristic, not a proven content/style separator. With ten
source calibration states, support estimates are weak and all confidence and
abstention values must be inspected.

## Rotation output trust region

A candidate action's SO(3) rotation is bounded relative to Frozen CoRP after
candidate sampling. Default \`--rotation-budget-rad 0.0\` preserves Frozen CoRP
rotations exactly. A positive budget allows a bounded geodesic deviation.
This controls perturbation magnitude; it does not establish positive task value.

## Validation scope

- Keep raw Mean and source-content-filtered Mean as ungated diagnostics.
- Add confidence-gated raw and content-filtered candidates.
- Add confidence-gated + rotation-trust-region candidates.
- Evaluate Clean-only estimators on their own Clean twin stream.
- Keep Shift-derived-correction-on-Clean as a separately labelled stress test.
- Aggregate all Clean controls over every record and per episode; never report
  the first decision state as a trajectory mean.
- This remains counterfactual action-distance validation only. No 20-episode
  or closed-loop success claim is authorized by this change.
