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

## PoNR result

Under OSMesa, tested failure states 43 and 45 had zero successful policy
branches at snapshots 0, 40 and 80. The harness now labels this as:

```text
ponr_step = 0
ponr_note = unrecoverable from step 0
```

This is a negative PoNR result: these failures are not recoverable by mere
chunk interruption, so no effective warning lead can be claimed for them.

## Recovery pilot (OSMesa)

Artifact v2, K=15, elastic interruption (executed_k=4 after alarm), no reflex.

State 36 (success):
- frozen 3/3; monitor 3/3; 0 alarms; 0 interventions.

State 43 (failure):
- frozen 0/3; monitor 0/3; 2 alarms / 2 interventions per seed at most;
- no rescue.

The monitor is safe on the tested success state and does not rescue this
hopeless failure. Recovery is therefore validated as non-harmful here, but not
as a success-rate improvement method.
