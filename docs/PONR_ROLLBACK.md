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
