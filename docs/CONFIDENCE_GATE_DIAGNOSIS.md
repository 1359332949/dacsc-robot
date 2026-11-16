# Confidence gate diagnosis and update-semantics repair

## V3 diagnosis

The V3 JSON only recorded the product confidence, so the residual and context
rejection mechanisms could not be distinguished. Reinspection showed that
the initial target observation in each of four episodes passed (confidence
about 0.70), while all later observations were rejected. Their residual norms
were large, suggesting the dynamic context-support term was the likely blocker.

There was also an update-semantics bug: the old implementation updated its EMA
with every observation, then multiplied the entire EMA by the *current*
observation confidence. A later OOD context therefore erased an earlier
accepted estimate and OOD residuals still polluted the EMA.

## Fix

- Log residual novelty and context support independently, including distances,
  thresholds, confidence values, abstention reasons, and acceptance.
- Use a fixed default minimum update confidence of 0.25.
- Only accepted observations update the episode EMA.
- Rejected observations neither update the EMA nor erase the last accepted
  correction.
- Scale the correction by the confidence associated with accepted updates,
  not by the most recent rejected observation.
- A Clean-like residual below the source OOF noise threshold does not initialize
  an estimate, preserving identity.

This explicitly assumes visual style is approximately stationary within an
episode. Later observations refine the estimate only when both source residual
novelty and context support pass. If later contexts are OOD, the first accepted
style estimate is held rather than extrapolating the source baseline.

## Required server validation

Re-run only the existing four held-out Object x Lighting 2222 trajectories x
three chunks, with identical seeds and all original baselines. Do not run
20 episodes.

Inspect per record:
- residual confidence versus context confidence;
- nearest context distance and source support radii;
- accept/reject reason;
- whether the accepted estimate is held on rejected chunks;
- correction norm before/after each decision;
- Clean-only correction cost aggregated over all records and per episode.

This is still counterfactual action-distance validation, not task-success evidence.
