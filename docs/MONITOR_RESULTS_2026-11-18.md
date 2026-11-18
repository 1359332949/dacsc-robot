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
