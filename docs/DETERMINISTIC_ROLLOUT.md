# Deterministic rollout protocol

## Root cause

EGL rendering was not deterministic across processes. Under `MUJOCO_GL=egl`,
the initial state, initial observation and first action chunk matched, but
chunk-2 observations/actions diverged; repeated identical state-35 runs gave
different success and step counts.

Under OSMesa:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

two identical runs produced identical chunk action hashes and identical step
counts. OSMesa is now required by `verify_ponr_rollback.py` and by
`monitor_recovery_eval.py` unless `--allow-nondeterministic-renderer` is
passed.

## Rollback sanity (OSMesa)

State 36 (target-dev shift, original success):

- original: success, 101 steps;
- full action replay from step 0: 2/2 success;
- policy branches from step 0: 2/2 success;
- branch 0 first-action diff: 0.0.

State 43 (target-dev shift, original failure):

- policy branches at steps 0/40/80: 0/3 total;
- original: failure, 220 steps.

## Shadow smoke

OSMesa, monitor artifact v2 (3-member ensemble, no visual, K=15):

State 36 success:
- frozen 3/3 success, 0 alarms;
- shadow 3/3 success, 0 alarms.

State 43 failure:
- frozen 0/3 success, 0 alarms;
- shadow 0/3 success, 4-6 alarms, 0 interventions.

This validates that the model can now be run deterministically in shadow mode.
It does not validate recovery or effective PoNR.
