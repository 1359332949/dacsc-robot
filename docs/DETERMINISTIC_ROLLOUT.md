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

## Monte-Carlo PoNR (branches=5, OSMesa)

The previous branches=1 result was exact replay, not a Monte-Carlo estimate.
With branches=5:

| state | step 0 | step 40 | step 80 | interpretation |
|---|---:|---:|---:|---|
| 43 | 0/5 | 0/5 | 0/5 | policy incompetence from start |
| 45 | 0/5 | 0/5 | 0/5 | policy incompetence from start |
| 34 | 0/5 | 0/5 | 0/5 | policy incompetence from start |
| 48 | 1/5 | 0/5 | 0/5 | **true PoNR between 0 and 40** |

State 48 is the first state with a real recoverability transition:
`P(success|s_0)=0.2`, `P(success|s_40)=0`.

## Effective lead check (state 48)

- PoNR candidate: step 40;
- monitor first alarm (seed 7777, shadow): step 112;
- alarm occurs after PoNR.

Therefore the monitor has **no effective lead time** on this true-PoNR state.
Recovery state 48: frozen 0/3, monitor 0/3, 2 interventions per seed; no rescue.
