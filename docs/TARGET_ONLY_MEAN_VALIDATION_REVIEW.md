# Target-only Mean validation review

## Findings in the previous action-distance script

The result archived as target_only_action_diag_clip20.json is retained as a
historical artifact, but its metric labels and later-trajectory protocol are not
valid as originally described.

1. normalized_mae was computed after unnormalize_action. It was therefore a
   controller-space mixed-dimension MAE, not normalized action MAE.
2. rot_mae was a component-wise absolute error on the 3D rotation vector, not a
   rotation geodesic.
3. The action sampler returns normalized CoRP actions. The script sent those
   normalized values directly to LIBERO instead of first calling
   normalizer.unnormalize_action. Thus, states after the first executed chunk did
   not follow the official Frozen CoRP execution protocol.
4. The Clean/Shift twin renderer did not explicitly fail closed on simulator
   physics incompatibility.
5. The same Shift environment object was reused across held-out episodes.

These issues do not invalidate the sampled action tensors at each recorded
state, but they invalidate the old metric label and the claimed multi-chunk
physical trajectory. Do not use the old aggregate table as the corrected
action-distance result.

## Corrections in validation v2

- Uses the official normalized-to-controller action conversion before
  executing Frozen CoRP actions.
- Creates a fresh Shift and Clean environment pair per held-out episode; does
  not call reset on the variant Shift environment.
- Requires Clean/Shift simulator physics compatibility and paired proprioception
  agreement before action comparison.
- Reports normalized action MAE before unnormalization, XYZ controller-unit MAE,
  rotation geodesic radians, and gripper MAE separately.
- Uses the same recorded flow-noise seed for all counterfactual methods at each
  decision state.
- Adds a reverse-direction control.
- Adds an experimental source-content residual projector fitted only from
  cross-fitted source-Clean residuals.
- Records raw target residual, source-Clean negative-control residual,
  projected residual, and retained-norm ratio separately.
- Keeps target Clean features outside all target estimator updates.

## Interpretation boundary

The source-content projector is a candidate filter, not a proven solution. It
assumes dominant out-of-fold residual directions on source-Clean data capture
content/baseline variation that should not be interpreted as a visual-style
offset. A genuine style shift may overlap those directions and be removed too.
The new action-distance diagnostic must determine whether the filter improves
geometric action metrics without harming Clean retention.

The 0.969 direction result is a one-observation-per-state direction check.
Since each state contributes exactly one update, EMA momentum does not
participate in that result. It is not evidence of temporal online estimation
over a trajectory.
