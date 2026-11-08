# V10 Atomic Target Pilot

## Identity

This is a custom CoRP-RoboCasa adaptation pilot, not an official CoRP
RoboTwin/RoboCasa reproduction and not an official RoboCasa leaderboard run.

The frozen machine-readable protocol is
`configs/v10_atomic_target_pilot.json`. The thin launchers read its defaults
through `python -m dacsc.v10_config --shell`; explicit environment overrides
remain visible in the resulting command and are recorded by the checkpoint or
SR result.

## Training tasks

```text
OpenDrawer
OpenCabinet
TurnOffStove
TurnOnSinkFaucet
TurnOnElectricKettle
```

All five are in the official RoboCasa target Atomic-Seen task list.

## Pretrain split

```text
train styles: 11-50
source holdout: residue 9 among styles 11-50
source pilot panel: styles 11-30 (20 episodes/task)
internal val: styles 51-60
```

The source holdout rows above define the offline training diagnostic. The
closed-loop source pilot uses a fixed 20-style panel (`11-30`) so that 20
episodes/task cover each source style once; it is a subset of the checkpoint's
training-style range and must not be reported as full `11-50` coverage. If full
coverage is required, source must use 40 episodes/task.

Closed-loop defaults are frozen in the same JSON:

```text
episodes/task: 20 for source, internal val and target
pretrain layout: explicit task-supported layout; no global default
action horizon: 10
executed action steps: 8
```

A style interval alone is not evidence that a task has samples in that panel.
Before every source evaluation, `scripts/check_eval_support.py` verifies that
the requested layout appears in that task's train split and that every
requested style has at least one train-split episode. It also records pairwise
layout/style support so sparse combinations are visible. A source panel without
task-level support fails before MuJoCo starts.

Target demonstrations are never used for training.

## Target evaluation

Target evaluation uses the official environment entry point:

```python
RoboCasaGymEnv(env_name=task, split="target")
```

The evaluator must not manually pass `style_ids=1-10` and
`layout_ids=1-10`; the official wrapper fixes the target scene pairs to
`(1,1), ..., (10,10)` and selects target object instances.

The frozen target schedule cycles through those ten official pairs. A target
manifest records `scene_split=target`, the requested/actual pair, target policy
seed, and simulator state, so every method replays the same scene and initial
state.

## Pilot budget

```text
steps: 30,000
batch: 96
lr: 5e-5
EMA: 0.999
grad clip: 1.0
precision: BF16
task aux: 0.01
seed: 42
```

This budget is a project pilot, not an official CoRP or RoboCasa budget.

Video caching is an execution option, not a new data path. `CACHE_VIDEO=1`
passes an explicit task allowlist to RoboCasa's official
`CachedLeRobotSingleDataset`; `CACHE_VIDEO=0` omits the cache and uses the
official DP streaming reader. Cache task names are checked against the selected
datasets, so a typo is a hard error. The frozen pilot caches the four tasks that
fit the server cgroup budget and keeps `OpenCabinet` on the streaming reader.
The read-only video validation compares the official `opencv` timestamp path
with the official `pyav` full-frame cache on real video; any pixel difference
fails the audit.

Every checkpoint records both the planned budget in `budget.steps` and the
actual checkpoint step in `training_origin.completed_steps`; the latter is part
of the SR comparison identity.

## Portable code identity

`scripts/pack_for_server.sh` adds `CODE_IDENTITY.json` to the exported archive.
It records the source commit and SHA256 for every archived source file. When a
training run starts without `.git`, the training entry verifies that manifest
before recording `project_git_commit`; a missing or tampered manifest is a hard
error.

## Gates

Model selection, budget selection, and method admission use only pretrain
source holdout/internal validation, never target outcomes.

1. The baseline must have nonzero clean internal-val success before method
   comparison.
2. Matched color-jitter/blur/camera/base shifts must show a measurable paired
   outcome degradation in internal validation; marginal success rates alone are
   insufficient.
3. Target is run only after methods, checkpoints, and hyperparameters are
   frozen. Target outcomes are external results, not a selection signal.
4. If target validation is zero or fails, report it as an external-transfer
   failure; do not tune on target and rerun it as if it were validation.

The current main research route is FFN-channel/token-channel analysis.
Its interventions must use their own specified unit-selection protocol and
matched random controls. A candidate must report clean/non-inferiority and the
shifted result against: no intervention, standard pixel augmentation, and a
random intervention matched on layer, view, token class, count, and edit
magnitude.

The retired statistical token-mixing implementation is not part of V10 and is
not an admissible fallback or comparison path.

## Entry points

Training:

```bash
OUT=/path/to/run \
PY=/path/to/robocasa/bin/python \
bash scripts/run_v10_pilot.sh
```

`PY` must be the RoboCasa environment because it reads RoboCasa365/DP videos
and train batches. The bootstrap installs the official CoRP package into that
same environment with `--no-deps`; do not use the separate NumPy 1.26.4 CoRP
LIBERO inference environment for training.

Target SR, with the baseline creating the frozen target manifests first:

```bash
MODEL=/path/to/model.pt \
OUT=/path/to/target_sr \
MANIFEST_ROOT=/path/to/target_manifests \
MANIFEST_MODE=create \
EPISODES=20 \
bash scripts/run_v10_target_sr.sh
```

Every later method must use `MANIFEST_MODE=replay` with the same
`MANIFEST_ROOT` and episode count.
