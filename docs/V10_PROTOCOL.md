# V10 Atomic Target Pilot

## Identity

This is a custom CoRP-RoboCasa adaptation pilot, not an official CoRP
RoboTwin/RoboCasa reproduction and not an official RoboCasa leaderboard run.

The frozen machine-readable protocol is
`configs/v10_atomic_target_pilot.json`.

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
internal val: styles 51-60
```

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
