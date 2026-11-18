# Runtime failure-monitor pipeline (implemented, not yet trained on real data)

## Inputs

Per-step trajectory JSONs from `scripts/trajectory_collector.py`:

- post-step `state8`;
- executed/clipped action;
- 768-D frozen visual feature.

No simulator object/contact truth is an input. Episode success is used only as a
training/evaluation label.

## Dataset

`scripts/build_monitor_dataset.py`:

- sliding window, default 10, stride 2;
- features: state8, clipped action, visual delta;
- labels: episode-level failure;
- group key: state_id (state-grouped split, no state leakage);
- refuses a one-class dataset.

## Model/training

`dacsc/failure_monitor.py` + `scripts/train_failure_monitor.py`:

- visual projection 768->32;
- 2-layer GRU;
- state-grouped validation, stratified to include failure and success groups;
- metrics: AUROC, Recall@5%FPR.

Synthetic smoke test passed: 4 episodes -> 44 windows, training 2 epochs,
AUROC 0.585, Recall@5%FPR 0.136. This is a pipeline test, not a result.

## Real-data blocker

The current real probe has only one per-step episode and no source/target
failure balance. Do not train or report monitor metrics until:

1. source fit/val full trajectories exist;
2. target-dev failures and successes exist;
3. state-grouped splits are frozen;
4. no simulator truth enters deployment features.

## Offline evaluation

`scripts/evaluate_monitor_offline.py` reports AUROC, Recall@5%FPR, first-alarm
remaining steps, median lead time, and false-alarm episodes per episode.

Synthetic smoke test (4 states, 44 windows, 2 epochs) produced finite metrics
and a median lead of 18 steps; this only exercises the code path.

## Closed-loop harness

`scripts/monitor_recovery_eval.py` loads a trained checkpoint, feeds the
deployable window, and can interrupt a chunk or run a retreat/open reflex.
It is implemented but must not be run until the monitor is trained and
validated on target dev.
