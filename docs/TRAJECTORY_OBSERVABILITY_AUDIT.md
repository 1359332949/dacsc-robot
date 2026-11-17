# Trajectory Observability Audit (2026-11-17)

Read-only audit of existing trajectories for a runtime failure monitor.

## Existing data

- 20 target-dev episodes: 10 states x Clean/Shift, seed 2014;
- 9/10 Clean success, 4/10 Shift success;
- episode length 81-220 steps (median 115.5);
- per-step executed action: available;
- per-chunk state8: available (7-19 entries per episode);
- per-step state8: not available;
- per-step RGB / frozen visual feature: not available;
- raw pre-clip action: not available;
- failure-event timestamp: not available;
- task phase: heuristic proxy only (no_lift / lift_failed / success);
- object/contact ground truth: not available.

## Split status

- source_fit: 10 states;
- source_validation: 10 states;
- target_dev: 10 states;
- target_test: 20 states, already used for exploratory comparisons.

## Conclusion

`monitor_training_ready = false`.

A failure monitor cannot be trained honestly on the current data. The minimum
missing pieces are:

1. per-step frozen visual features (or RGB);
2. per-step state and executed action, with chunk/replan index;
3. raw vs clipped actions;
4. failure-event time or a well-defined post-hoc label window;
5. verified phase labels or object/contact truth for supervision;
6. recovery-interface validation;
7. a fresh target test split.

The existing first-chunk pairs are **known-perturbation paired calibration**,
not target-only TTA data, and cover only chunk 0.

Next step is data collection and schema validation, not verifier training.
