# SPACA v1: Source-Paired Action-Consistent Adaptation

## Claim boundary

SPACA v1 is **paired-domain supervised behavioral calibration**, not
target-only unsupervised TTA. It uses source clean/shift pairs from the same
physical state, the same explicit flow-matching z0, and a frozen CoRP.

## Components

- frozen DINOv2 / Resampler / language / state / DiT;
- zero-initialized residual adapter on the 96x768 visual tokens;
- paired normalized-action loss (shift -> clean);
- clean action-preservation loss;
- no teacher, no feature residual loss, no smoothness loss in v1.

## Pair integrity

Pair collection uses full MuJoCo `get_state/set_state`, controller state,
episode flags, OSMesa, and explicit z0. Source-fit states 0/1 are train;
source-fit state 2 is held out. Target dev is evaluation only; target test is
never used.

## Offline held-out action error (source state 2)

| mode | RMSE | XYZ | Gripper |
|---|---:|---:|---:|
| frozen | 0.0687-0.0767 | 0.1017-0.1153 | 0.00151-0.00161 |
| adapter | 0.0444-0.0620 | 0.0636-0.0857 | 0.00178-0.00195 |

Held-out state 2 improves in overall/XYZ error but not gripper RMSE.

## Closed-loop target dev (OSMesa, same z0)

| state/domain/seed | frozen | adapter |
|---|---:|---:|
| 36 shift 7777 | 1/1 | 1/1 |
| 36 clean 7777 | 1/1 | 1/1 |
| 43 shift 7777 | 0/1 | **1/1** |
| 43 shift 8888 | 0/1 | **1/1** |
| 43 shift 9999 | 0/1 | **1/1** |
| 45 shift 7777 | 0/1 | 0/1 |

This is a pilot: 2 training states, 6 pair records, one adapter seed, and only
target-dev evaluation. Do not claim general SR improvement or target-test
generalization.

## Expanded source pairs (v1b)

Source fit states 0,1,2,4,6,8; two chunks each; 12 records, 11 usable.
Train states 0,1,2,4,6; validation state 8.

Offline held-out (state 8) summary:

- frozen RMSE 0.0721, XYZ 0.0964, gripper 0.00204;
- adapter RMSE 0.0223, XYZ 0.0227, gripper 0.00251.

The adapter improves overall/XYZ but **hurts gripper RMSE** on held-out state 8.
Closed-loop state 43 shift: frozen 0/1, adapter 1/1 (seed 7777).
State 45 shift: frozen 0/1, adapter 0/1.

Do not claim a general SR improvement. The gripper regression and state-45
failure are the main remaining defects.

## Gripper-weighted variant

Training uses per-dimension Huber weights with gripper weight 20.

Held-out state 8:

- frozen RMSE 0.0721, XYZ 0.0964, gripper 0.00204;
- adapter RMSE 0.0227, XYZ 0.0243, gripper 0.00143.

Gripper RMSE improves without losing overall/XYZ. Closed-loop state 43 still
0/1 -> 1/1; state 45 remains 0/1 -> 0/1.

This is the current best v1 adapter, but still target-dev pilot only.

## Full target-dev adapter-only failure and gate status

Adapter-only closed loop on target-dev shift (seed 7777):

| state | frozen | adapter |
|---|---:|---:|
| 34 | 0 | 0 |
| 35 | 0 | 0 |
| 36 | 1 | **0** |
| 37 | 1 | **0** |
| 39 | 0 | 0 |
| 43 | 0 | **1** |
| 45 | 0 | 0 |

Adapter-only is therefore not deployable: it rescues state 43 but breaks
states 36 and 37.

A monitor-gated variant was implemented. On state 36 it correctly kept the
frozen path (success), but on state 43 the monitor gate did not activate
(gate_used=false), so state 43 remained failure. The monitor gate needs a
separate calibration/debug pass before any full target-dev run.

Do not start the full target-dev adapter evaluation until the gate fix is
validated.

## Gate experiments (negative)

Adapter-only rescues state 43 but breaks states 36/37. Gate diagnostics:

- monitor persistence 15: gate activates at step 54; state 43 still fails;
- monitor persistence 1: gate activates at step 26; state 43 still fails;
- state 36 monitor max prob is 0.243, so gate correctly stays off;
- adapter-first-chunk preserves state 36/37 (both success) but does not rescue
  state 43.

Therefore no current gate solves both success preservation and recovery.
Adapter-only is not deployable. Do not start full target-dev / 20-episode.

## Phase 0 strict pairing audit (completed)

The pair collector is now fail-closed:

- strict `model_compatible` with body/geometry/physics checks;
- full `compare()` of qpos/qvel/act/ctrl/body/EEF (derived body/EEF
  tolerance 1e-3);
- camera geometry `camera_compare(expect_equal=True, tol=1e-3)`;
- snapshot digests and model/state/camera pass flags in every record;
- no swallowed observable exception.

Final source-fit collection (states 0,1,2,4,6,8; 2 chunks each):

```text
12 records
11 pairing PASS
1 record excluded: state 8 chunk 1, state8_max=1.18e-3
```

The earlier 11 pairs are provisionally invalid for strict claims; they predate
the full model/state/camera audit.

Gate diagnostics:

- `gate_step` is the monitor trigger time, not the adapter application time;
- adapter actually applies at the next replan boundary;
- monitor persistence 1 triggers at step 26; state 43 still fails;
- first-chunk adapter preserves 36/37 but does not rescue 43.

No adapter retraining or full target-dev rollout was performed in Phase 0.

## Strict multi-z0 adapter (v1c)

Data: final strict pairs (11 pass; state 8 chunk 1 excluded).
Training: pair + identity + z0 augmentation (4 z0) + residual 0.01;
train states 0,1,2,4,6; validation state 8.

Closed-loop target-dev shift, seed 7777:

| state | frozen | adapter |
|---|---:|---:|
| 36 | 1 | 1 |
| 37 | 1 | 1 |
| 43 | 0 | 1 |
| 45 | 0 | 1 |

Clean target-dev:

| state | frozen | adapter |
|---|---:|---:|
| 36 | 1 | 1 |
| 37 | 1 | 1 |
| 43 | 1 | 1 |

But multi-seed is unstable:

- state43: adapter 1/1 at seed7777, 0/1 at seed8888, 0/1 at seed9999;
- state45: adapter 1/1 at seed8888; seed9999 pending/not robust.

Conclusion: multi-z0 training removed the 36/37 breakage in this seed and
gives a strong seed-7777 pilot, but the recovery is still seed-sensitive.
Do not claim general SR improvement or start full target-dev confirmation.

## Phase 1A stochasticity attribution

Rollout provenance now records `env_seed`, `z0_seed`, z0 hash, adapter
checkpoint hash, adapter forward calls, applied chunks and first application
step. Adapter rollouts show `adapter_forward_calls > 0` and
`adapter_first_applied_step = 0`, so they are not silent frozen runs.

### State 43, fixed env seed 7777, varying z0 seed

| z0 seed | frozen | adapter |
|---|---:|---:|
| 7777 | 0 | 1 |
| 8888 | 0 | 1 |
| 9999 | 0 | 1 |
| 1111 | 0 | 0 |
| 2222 | 0 | 1 |

Adapter 4/5, frozen 0/5. Adapter recovery is z0-sensitive.

### State 43, fixed z0 seed 7777, varying env seed

| env seed | frozen | adapter |
|---|---:|---:|
| 7777 | 0 | 1 |
| 8888 | 0 | 1 |
| 9999 | 0 | 0 |
| 1111 | 0 | 1 |

Adapter 3/4, frozen 0/4. Adapter recovery is also env-sensitive.

### Successful states 36/37, fixed env seed, varying z0 seed

At z0=7777 adapter preserves both states. At z0=8888 and 9999 the adapter
breaks state 36; state 37 also breaks at z0=8888. Thus the adapter is not a
stable always-on method and cannot be released to full target-dev.

Target test remains untouched.

### Completed z0 sweep on successful states 36/37

Fixed env seed 7777:

| state | z0 | frozen | adapter |
|---|---|---:|---:|
| 36 | 7777 | 1 | 1 |
| 36 | 8888 | 1 | 0 |
| 36 | 9999 | 1 | 0 |
| 37 | 7777 | 1 | 1 |
| 37 | 8888 | 1 | 0 |
| 37 | 9999 | 0 | 1 |

Frozen 5/6, adapter 3/6. The adapter both breaks successes and rescues a
frozen failure at a different z0. This confirms that the current adapter is
not an always-on improvement and that z0 sensitivity is bidirectional.

## Adapter support diagnostics (state-level, fixed env seed 7777)

For states 36/37/43/45 and z0 seeds 7777/8888/9999:

- residual ratio (mean over two views): 0.124-0.146 for all states;
- action delta RMSE: 0.042-0.058;
- action delta gripper: 0.0016-0.0037.

There is no simple separation between successful and unsuccessful states or
between z0 seeds. Residual magnitude alone is not an adapter-benefit signal.

Combined with the z0 sweep:

- state 43: adapter 4/5, frozen 0/5;
- states 36/37: frozen 5/6, adapter 3/6;
- adapter breaks successes and rescues one frozen failure.

Conclusion: a residual/action-delta threshold gate is not supported by these
data. The next measurable question is adapter training variance (independent
adapter seeds), not another heuristic gate.

## Critical correction: clean proprio token bug and superseded results

The collector previously saved only the Shift state token. Clean identity and
the Clean Frozen target therefore used the Shift proprio token. This invalidated
the earlier seed-7777 4/4 result and the earlier stochasticity numbers.

Fixes:

- collector saves `shift_state_token` and `clean_state_token`;
- trainer uses the Clean token for Clean identity/target;
- trainer adds a parity check: recomputed Clean Frozen action must match the
  stored Clean action (`clean_action_parity_max = 0.0`);
- NumPy import fixed;
- controller audit is fail-closed for missing/mismatched controller state;
- rollout logs separate env seed, z0 seed, all-z0 hashes, adapter apply counts,
  clipping and normalized/controller deltas.

Corrected strict adapter (train states 0,1,2,4,6; validation 8; z0 augment 4;
grip weight 20; residual 0.01), target-dev shift, env seed 7777, z0 seed 7777:

| state | frozen | corrected adapter |
|---|---:|---:|
| 36 | 1 | 0 |
| 37 | 1 | 1 |
| 43 | 0 | 1 |
| 45 | 0 | 0 |

Frozen 2/4, corrected adapter 2/4. The adapter breaks state 36 and rescues
state 43, with no net gain. All earlier SPACA seed-7777 4/4 and stochasticity
claims are superseded by this corrected checkpoint.

## Clipping-aware variant and final current checkpoint

After adding clean/shift proprio-token separation and parity (=0.0), the
corrected adapter is:

| state | frozen | corrected adapter | clip-aware adapter |
|---|---:|---:|---:|
| 36 | 1 | 0 | 1 |
| 37 | 1 | 1 | 0 |
| 43 | 0 | 1 | 0 |
| 45 | 0 | 0 | 0 |
| total | 2/4 | 2/4 | 1/4 |

The clipping-aware loss did not improve the protocol. The current checkpoint
has no net advantage over Frozen and must not be used for full target-dev.

Rollout logs now include per-chunk z0 hashes, normalized/controller/clipped
action deltas, clipping ratios, init state/model/controller digests and
adapter application counts. This proves the adapter is actually applied but
does not establish stable benefit.

## Clean Frozen attribution baseline (critical)

Corrected adapter on target-dev Clean, env seed 7777, z0 seed 7777:

| state | Clean Frozen | Clean Adapter |
|---|---:|---:|
| 36 | 1 | 1 |
| 37 | 1 | 1 |
| 43 | 1 | 0 |
| 45 | 1 | 1 |
| total | 4/4 | 3/4 |

Clean Frozen succeeds on all four target states; the adapter breaks Clean
state 43. Thus the failure is not explained by "Clean CoRP itself fails".
The adapter also harms the target Clean distribution, not only Shift.

Combined shift result: Frozen 2/4, corrected adapter 2/4. No net gain, and
Clean retention is not perfect. The always-on v1 adapter is not ready for
full target-dev.

## Full source-data adapter (10 train states, 10 validation states)

Data: 20 source-fit records, 20 source-validation records; 37 pairing-pass
records total. Training used fit-only, validation-only for checkpoint
selection; best epoch = 6 by validation loss.

Target-dev shift, env seed 7777, z0 seed 7777:

| state | frozen | full-data adapter |
|---|---:|---:|
| 36 | 1 | 0 |
| 37 | 1 | 1 |
| 43 | 0 | 0 |
| 45 | 0 | 1 |
| total | 2/4 | 1/4 |

Clean Frozen earlier: 4/4. The full-data adapter is worse than the small-data
adapter and still damages state 36. Expanding data alone did not fix the
always-on instability. No full target-dev run.

## Full-source validation and attribution

Source-validation action error (20 held-out records):

| mode | RMSE | XYZ | gripper |
|---|---:|---:|---:|
| frozen | 0.0803 | 0.1140 | 0.00175 |
| full-data adapter | 0.0451 | 0.0574 | 0.00171 |

Offline action error improves on held-out source-validation, but target-dev
closed-loop is worse (frozen 2/4, adapter 1/4). This is direct evidence that
source action consistency is not sufficient for closed-loop benefit.

Residual support (env seed 7777, three z0 seeds):

| domain | residual ratio range |
|---|---:|
| target Clean | 0.023-0.028 |
| target Shift | 0.093-0.104 |

Clean and Shift residual magnitudes are separable, but successful Shift state
36 and failure state 43 both have residual ~0.10. A residual gate can protect
Clean, but cannot distinguish beneficial Shift interventions.

Adapter reduces multi-z0 dispersion slightly (ratio 0.84-0.97); this does not
predict which Shift state benefits.

## Gripper-pair audit and clean residual bypass

Pair gripper audit (controller actions):

| split | records | max abs diff | >0.05 | >0.1 |
|---|---:|---:|---:|---:|
| source_fit | 18 | 0.00857 | 0 | 0 |
| source_validation | 19 | 0.00824 | 0 | 0 |

The paired Clean/Shift gripper actions are almost identical. The adapter
therefore has no training signal for the 84-96-step grasp-closure failure
mode; it can only fit XYZ residuals.

Clean residual bypass:

- if `residual_ratio < 0.05`, execute Frozen instead of Adapter;
- Clean target-dev: Frozen 4/4, Adapter 4/4;
- Shift target-dev: Frozen 2/4, Adapter 1/4 (state 45 only).

The bypass protects Clean but does not create Shift benefit.

Version attribution: the earlier stochasticity tables were produced with the
pre-clean-token `spaca_train_multi_z` checkpoint and are superseded. Current
numbers use `spaca_train_full` (clean-token corrected, best epoch 6).
