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
