# P0 protocol fixes (2026-11-17)

## Fixed

1. Formal runner now fails closed without a split:
   - `--split-json` and `--role` are required;
   - the old no-split path requires explicit `--allow-legacy-diagnostic`.
2. `target_test` is blocked unless the split is genuinely confirmatory:
   - `confirmatory_test=true`;
   - `target_test_used_for` is empty;
   - `target_test_status == locked_unused`.
   The current `libero_spatial` split is `exploratory_accessed`, so its
   `target_test` role cannot be run.
3. Seed semantics are role-based:
   - `target_dev` uses `screening_seeds`;
   - `target_test` uses `evaluation_seeds`;
   - source roles use `evaluation_seeds` and are not used for target claims.
4. Source capture can read the split JSON directly and records checkpoint,
   matrix, initial-state, split, and script hashes. The runner rejects a
   capture whose checkpoint/split hash does not match.
5. Result payload records `git_commit`, script hash, matrix/split/capture/BDDL/
   initial-state hashes, and the seedpool role.
6. CI compiles `dacsc/tta_protocol.py` and runs `tests.test_tta_protocol`.

## Not fixed yet

- The split still has only one Lighting ID for both target dev and target test.
  It measures within-known-shift state generalization, not unseen perturbation
  generalization.
- The canonical branch is not unified with
  `exp/v5-per-observation-residual-gate`.
- CoRP training-state exposure is not independently verified.
- The target test set is exploratory and cannot support a confirmatory claim.
- V5 results remain action-distance diagnostics, not closed-loop SR.
- Full perturbation-level dev/test split is not materialized.

## Consequences

The current `libero_spatial` split is a pilot only. Any target-dev run now uses
screening seeds, so previous target-dev numbers collected with seed 2014 are
legacy/exploratory and must not be mixed with protocol-compliant runs.

## Perturbation-level pilot split (2026-11-17)

New pilot split: `docs/data/tta_split_libero_spatial_perturbation.json`

- source: Clean;
- target development: Lighting 2292, 10 dev states;
- target test: Lighting 2312, 20 heldout states;
- perturbation config 2312 was never used before;
- target test states were used under 2292 exploratory test, so this is not a
  fully fresh state test; it is a fresh perturbation configuration.

The split status is `admission_accessed` after a Frozen-only baseline run. It
is not confirmatory and cannot be used for method selection.

Protocol-compliant target-dev Frozen (screening seeds 7777/8888/9999):
11/30.

Target-test admission on Lighting 2312, Frozen, seed 2014:
Clean 18/20, Shift 1/20.

These are admission/baseline numbers, not a TTA result.
