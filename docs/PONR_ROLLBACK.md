# Point-of-No-Return rollback probe

Motivation: failure lead time must be measured against the first
irrecoverable state, not episode timeout.

`scripts/verify_ponr_rollback.py`:

1. runs a frozen rollout and saves MuJoCo snapshots at requested steps;
2. restores each snapshot in a fresh environment;
3. runs N frozen-policy branches;
4. reports branch success rates and a candidate PoNR step.

Protocol guardrails:

- target-dev only; target test is not touched;
- model compatibility is checked before restore;
- branch seeds are recorded;
- no recovery action is inserted during branching.

## Preliminary probe (state 34, target-dev shift, seed 7777)

Snapshots at steps 20, 40, 80, 100, 120 all produced 0/1 or 0/2 successful
branches. This cannot yet be interpreted as a PoNR: the rollback/render path
needs validation (controller state and observation freshness), and failure may
start before step 20. Do not report these numbers as effective warning lead.

## Controller/observation consistency status

Follow-up fixes attempted:

- snapshot OSC controller `goal_pos`, `goal_ori`, `ori_ref`, `relative_ori`,
  and interpolator `start/goal/step`;
- restore controller state with the MuJoCo state;
- save the exact observation at the snapshot and reuse it for the first
  branch action;
- restore in the same simulator instead of a fresh environment;
- call `sim.forward()` after restore.

Despite these, a successful state-35 branch at step 0 still returned 0/1
success while the original rollout succeeded. The rollback harness is therefore
**not yet validated**; do not use its branch success rates as PoNR evidence.
Next check: compare the first post-restore action against the original action,
and restore any remaining robot/controller buffers.

## Additional rollback fixes and test

Added:

- robosuite episode `done`/`timestep` reset after restore;
- exact observation at snapshot;
- RNG state (torch/numpy/cuda) at snapshot;
- exact-RNG branch for branch 0.

Sanity test (state 35, shift, success):

- snapshot 0: 1/2 branches success;
- snapshot 20: 0/2 branches success.

The exact-RNG branch still differs from the original first action by about
0.14, so the harness is not exact yet. Do not interpret the step-20 drop as a
validated PoNR; it is exploratory.
