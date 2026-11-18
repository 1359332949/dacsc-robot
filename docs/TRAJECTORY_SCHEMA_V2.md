# Trajectory schema v2 (per-step probe)

`scripts/trajectory_collector.py` now records, for every executed action step:

- `state8_pre` and `state8` (after action);
- `visual_pre` and `visual_mean` (768-D frozen Resampler mean, before/after);
- `raw_action` and `clipped_action`;
- `reward`, `done`, `success`;
- `chunk`, `step`;
- key frames every N steps.

Schema probe: target dev state 34, Lighting 2292, seed 7777, 60 steps.

- 60 rows, 8-D state, 768-D visual, 7-D action;
- `state8_pre != state8` for all 60 steps;
- visual variance nonzero;
- 5 action chunks.

Still missing for a deployable monitor:

- failure-event timestamp (only final success/timeout);
- verified task phase / object / contact labels;
- full source trajectory set and independent source validation;
- recovery-interface validation.

Old chunk-level trajectories are superseded and must not be used for per-step
monitor training.
