# Runtime failure monitor: first real-data run (2026-11-18)

Data: source fit+validation (40 episodes: Clean/Shift), target dev (20
episodes: Clean/Shift), one seed 7777, Lighting 2292, frozen CoRP.

## Data balance

- source fit: Clean 10/10 success, Shift 3/10 success;
- source validation: Clean 10/10, Shift 4/10;
- target dev: Clean 8/10, Shift 7/10.

The monitor was trained on source only and evaluated on target dev only.

## Source-validation (training-time)

- AUROC 0.8785;
- Recall@5%FPR 0.514;
- 471 windows, 45% positive.

## Target-dev evaluation

- AUROC 0.8278;
- Recall@5%FPR 0.500;
- 1255 windows; 5 failure episodes, 15 success episodes;
- median first-alarm lead: 180 steps;
- **false alarms: 7/15 success episodes**.

## Interpretation

The monitor has discriminative signal (AUC 0.83) but the target false-alarm
rate is too high for closed-loop intervention. It is a promising first signal,
not a deployable monitor.

Do not run chunk interruption/reflex from this checkpoint. Do not tune the
threshold on target test; any threshold selection must use source validation or
target dev as development data.

## Caveats

- one seed, 10 target-dev states;
- episode-level labels, no failure-event timestamp;
- source/target lighting 2292 only;
- source shift trajectories are used for failure examples (known-perturbation
  calibration);
- target test untouched but not confirmatory.

## Domain-confound ablation (same data)

The frozen monitor's input features classify Clean vs Shift almost perfectly:
domain AUC 0.982 on target dev. This indicates a domain shortcut.

Ablations:

| monitor | target AUC | source-fixed-threshold recall | source-fixed false-alarm episodes |
|---|---:|---:|---:|
| full (state+action+visual) | 0.828 | 0.655 | 10/15 |
| no visual, state+action | 0.830 | 0.670 | 4/15 |
| no visual + domain-adversarial | 0.829 | 0.674 | 6/15 |

Removing the visual feature cuts false alarms from 10/15 to 4/15, but the
domain classifier on state+action still reaches 0.872 AUC. The monitor is not
deployment-ready, and the target threshold must be fixed from source
validation, not target test.

Do not run recovery from these checkpoints.

## Next experiment (ready, not run)

The next monitor run should use relative state features:

```text
state_delta, action, visual_delta
```

and fixed source-validation threshold. `scripts/build_monitor_dataset.py`
supports `--state-mode delta`; `scripts/run_monitor_pipeline.py` supports
`--state-mode delta --no-visual`, and `evaluate_monitor_offline.py` supports
`--source` to fix the threshold from source validation rather than target test.

Required comparison:

- full monitor (visual, absolute state);
- no-visual monitor;
- no-visual + state-delta;
- threshold fixed on source validation.

Do not select a variant by target test.

## Grouping correction (important)

The first monitor run combined `source_fit` and `source_validation` and then
randomly held out state groups. That violates the split roles and invalidates
the earlier ablation numbers as role-correct evidence.

New role-correct pipeline:

- train on `source_fit` only;
- validate/threshold on `source_validation` only;
- evaluate on `target_dev` only;
- target test untouched.

Role-correct full monitor:

- source-validation AUROC 0.930;
- target-dev AUROC 0.852;
- source-fixed recall 0.689;
- source-fixed false alarms **10/15**;
- median lead 204 steps.

This is still not deployable. The earlier no-visual/domain-adversarial
ablations must be rerun with this role-correct pipeline before being cited.

Data audit: source_fit/source_validation/target_dev each 20 episodes; state
sets disjoint; no target-test state in target_dev; per-step dims valid
(state8/action7/visual768). No audit errors.

## Role-correct ablation (source_fit train, source_validation threshold)

| monitor | source-val AUC | target AUC | source-fixed recall | false alarms |
|---|---:|---:|---:|---:|
| full, absolute state | 0.930 | 0.852 | 0.689 | 10/15 |
| no visual, absolute state | **0.952** | **0.873** | **0.742** | **4/15** |
| no visual, state-delta | 0.893 | 0.842 | 0.509 | 5/15 |
| full, state-delta | 0.918 | 0.812 | 0.592 | 11/15 |

Best role-correct variant: **no visual + absolute state** (AUC 0.873,
recall 0.742, false alarms 4/15). It is still not deployment-ready; do not run
recovery.

This also shows state-delta is worse than absolute state in this pilot; the
domain shortcut is not solved by relative state alone.

## False-alarm diagnosis (best role-correct variant)

Best variant: no-visual + absolute state, source-fixed threshold.

- target AUC 0.873, recall 0.742, false alarms 4/15;
- false alarms: 3 shift successes (states 35, 44, 45) alarm early (steps
  35-37), and 1 clean success (state 49) alarms late (step 113).

So the remaining errors are mostly early Shift successes, consistent with a
residual domain shortcut rather than a pure failure signal.

## Additional role-correct variants

| monitor | target AUC | recall | false alarms |
|---|---:|---:|---:|
| no-visual + absolute (best) | 0.873 | 0.742 | 4/15 |
| no-visual + domain-adversarial | 0.864 | 0.717 | 5/15 |
| within-horizon labels (H=10) | 0.648 | 0.400 | 3/15 |

Within-horizon labels reduce false alarms but lose too much failure
discrimination (AUC 0.648, median lead 62). Domain-adversarial is not better
than the no-visual baseline. No recovery run.

## Temporal persistence (source-selected)

Post-processing: require K consecutive windows above the source-fixed threshold.

| K | source false alarms | source failures detected | target false alarms | target failures detected | target median lead |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 6/6 | 4/15 | 5/5 | 192 |
| 5 | 3 | 6/6 | 4/15 | 5/5 | 184 |
| 10 | 0 | 6/6 | 2/15 | 5/5 | 174 |
| 15 | 0 | 6/6 | 1/15 | 5/5 | 164 |
| **20** | **0** | **6/6** | **0/15** | **5/5** | **154** |

K=20 is selected on source validation (zero source false alarms, all source
failures detected), not on target test. On target dev it yields zero false
alarms and detects all five failures. This is a strong pilot result, but it is
still one seed, one perturbation condition, and 10 target-dev states; do not
run recovery or claim deployment yet.

## Multi-seed validation

Data: seed 7777 + 8888; source_fit/val and target_dev combined.

Monitor: no visual + absolute state; trained on source_fit; source-val fixed
threshold; K selected on source validation.

At K=15:

- source validation: 15/15 failures detected, 1 false alarm;
- target dev: 13/13 failures detected, 1/27 success false alarm;
- target median lead: 164 steps;
- target AUC (window-level): 0.922.

At K=30: target 10/13 failures, 0/27 false alarms. K=15 is the better
source-selected operating point in this pilot.

This is multi-seed evidence, but still one known perturbation condition
(Lighting 2292), target test untouched, and only 10 target-dev states.
Do not run recovery yet.
