# Confidence-gate diagnosis and update/application repair

## V3 diagnosis

V3 logged only the product confidence, preventing residual novelty and
proprioceptive context support from being distinguished. The initial target
observation in each of four episodes passed (confidence about 0.70), while
later observations were rejected. Large later residual norms suggested that
context support, rather than residual novelty, was the likely blocker.

The old update path also allowed every observation to pollute the EMA and then
scaled that EMA by the current observation confidence. A later OOD context
could erase a previously accepted estimate.

## V4 repair and result

V4 separated residual and context diagnostics, accepted only observations
above the source-OOF calibration confidence maximum, and held the last accepted
episode estimate when a later observation was rejected. The default threshold
is fitted from source-Clean OOF data; it is not a fixed 0.25. In the reported
Object x Lighting 2222 run it was 0.03548.

V4 fixed the update-state bug but exposed an application-time gap: the stored
episode correction was applied to every current input. The paired-Clean
transfer stress test therefore measured the cost of unconditional transfer,
not whether an observation-local gate could preserve Clean behavior.

The V4 counterfactual action-distance result is not closed-loop success-rate
evidence. The 20-episode screen remains unauthorized until the mechanism
diagnostics meet both Target adaptation and Clean-retention criteria.

## V5 discrimination experiment

V5 keeps the update gate unchanged:

- Update acceptance = source residual novelty x source-context support.
- Rejected observations do not update the EMA or erase the last accepted
  estimate.
- Application confidence = the current observation's source residual novelty
  only. Context support is not reused at application time because later valid
  task states can move outside source-state support.

For each paired Shift/Clean simulator state, the diagnostic evaluates:

1. Target correction with the original unconditional V4 application.
2. Target correction scaled by the current Shift residual-novelty confidence.
3. The same stored Target correction scaled by the paired Clean residual-
   novelty confidence (offline negative control only).
4. The corresponding content-filtered variants.

The Clean stream is used only for offline counterfactual evaluation. At
deployment, application confidence is computed from the current observation
available to the policy; no paired Clean observation, reward, success label,
or simulator state is provided to the estimator.

Interpretation:

- If Shift residual confidence remains positive on later chunks while Clean
  residual confidence is near zero, and the new Target action metrics remain
  better than Frozen while Clean transfer cost falls, the V4 failure was
  primarily unconditional application.
- If Shift and Clean residual confidence overlap, this confidence statistic
  does not discriminate style from content. Do not tune a threshold on held-out
  target/Clean pairs; return to source-only calibration or stop this gate.
- If application gating suppresses both Target and Clean, it is not a useful
  adaptation mechanism even if Clean retention looks good.

## Execution boundary

Run only the existing four held-out Object x Lighting 2222 trajectories x
three chunks with identical seeds and original baselines. Do not run 20
episodes or closed-loop success evaluation at this stage.

This remains counterfactual action-distance validation, not task-success
evidence.
