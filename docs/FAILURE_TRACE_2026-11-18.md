# Failure trace: Lighting 2312, state 3 vs success state 40

One Frozen episode per state, seed 2014, official `policy.predict`, 12-action
chunks, 220-step budget. Diagnostic only.

| state | result | steps | first gripper close | eef z min | final eef z |
|---|---|---|---:|---:|---:|
| 3 | fail (timeout) | 220 | ~96 | 0.928 | ~1.176 |
| 40 | success | 132 | ~84 | 0.945 | ~0.958 |

Failure state 3 descends, closes the gripper later than the successful state,
never lifts/places, then retreats to the starting height; it times out.
The images show the arm obscuring much of the agent view; the wrist view is
more informative.

Artifacts:

- `docs/figures/failure_trace_3/trace_plot.png`
- `docs/figures/failure_trace_3/frames_montage.png`
- `docs/figures/failure_trace_40/trace_plot.png`
- `docs/figures/failure_trace_40/frames_montage.png`

Caveat: this is a single seed and one failure/success pair. It localizes the
failure to the grasp/lift transition, not a full causal explanation.
