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

## Gates

1. The baseline must have nonzero clean target success before method comparison.
2. Matched color-jitter/blur/camera/base shifts must show a measurable paired
   outcome degradation; marginal success rates alone are insufficient.
3. `random_stats` is a mandatory control.
4. `cross_style_stats` must exceed the random control on the shifted condition
   without an unacceptable clean/non-inferiority loss.
5. Action-aware is not resumed until cross-style passes.
6. Test/target results must never be used for method selection or tuning.

## Entry points

Training:

```bash
OUT=/path/to/run \
PY=/path/to/python \
bash scripts/run_v10_pilot.sh
```

Target SR, with the baseline creating the frozen target manifests first:

```bash
MODEL=/path/to/model.pt \
OUT=/path/to/target_sr \
MANIFEST_ROOT=/path/to/target_manifests \
MANIFEST_MODE=create \
EPISODES=10 \
bash scripts/run_v10_target_sr.sh
```

Every later method must use `MANIFEST_MODE=replay` with the same
`MANIFEST_ROOT` and episode count.
