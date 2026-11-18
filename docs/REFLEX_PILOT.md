# Proprioceptive grasp-reflex pilot (target dev, 2026-11-18)

Rule-based diagnostic: after the gripper closes, if eef z does not rise by 1 cm
within 8 steps, either (a) interrupt the current chunk only, or (b) open the
gripper, retreat, and replan.

Target dev, Lighting 2292, seed 7777, 10 states:

| mode | success | interventions |
|---|---:|---:|
| Frozen | 7/10 | 0 |
| chunk interrupt only | 6/10 | — |
| reflex retreat/open/replan | 5/10 | — |

Both interruption and reflex are negative in this pilot. The detector can fire,
but the recovery action is not sufficient; do not tune thresholds or add a
learned monitor until trajectory data and failure labels are complete.

This is a diagnostic, not a method result.
