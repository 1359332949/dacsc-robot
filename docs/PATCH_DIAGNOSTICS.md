# CoRP DINO Patch Diagnostics (Phase 0, Phase 1, B0-B3)

## Claim boundary

This work is an **offline known-perturbation paired-simulator diagnostic**, not
a deployable TTA method. The state IDs come only from source_fit and
source_validation, but the Shift branch explicitly renders the known target
lighting configuration 2292 on those source initial states. Therefore this is
not source-free adaptation and does not establish generalization to an unseen
visual style.

Target-dev and target-test initial-state IDs, success labels, rewards, and
Clean twins are not used. Target-dev/Test outcomes are not used for model or
threshold selection. The Clean patches used by B2/B3 are privileged
counterfactual inputs for offline mediator interventions, never deployment
signals.

## Why a new collector is required

The existing early/late collectors save post-Resampler visual tokens. Those
files cannot establish whether DINO patch tokens changed; substituting them
would intervene at the wrong representation layer. The new collector records
the per-view visual_encoder output sequence before CrossAttentionResampler,
with the paired state inputs and explicit shared z0. If Phase 0 finds prefix
tokens, the collector preserves them because that is the exact sequence the
frozen Resampler consumes; in that case B0-B3 are full visual-token-sequence
replacement interventions, not patch-only interventions.

Patch token count and embedding width are discovered at runtime. There is no
assumption of 256 tokens, 384 channels, or a fixed number of views.

## Phase 0: verify the live interface

~~~bash
MUJOCO_GL=osmesa python scripts/verify_dino_patch_interface.py \
  --matrix /path/to/libero_matrix.json \
  --checkpoint repos/CoRP/checkpoints/corp_libero.pt \
  --split-json docs/data/tta_split_libero_spatial.json \
  --suite libero_spatial --lighting-id 2292 \
  --out runs/patch_diag/phase0_interface.json
~~~

The script selects a Clean source_fit initial state unless state-id is
provided. It records the preprocessed image shape, visual-encoder input/output
shape, per-view patch shape, model metadata candidates, possible patch-grid
matches, and identity-path errors at four levels:

- patch tensor;
- Resampler visual tokens;
- complete VLA representation tokens;
- Flow-Matching action with a fixed explicit z0.

**Gate:** all identity-path errors must be below 1e-6. This includes pinning
the official policy.predict path to the same explicit z0 and comparing its
controller-space action with the direct sampler. It also requires a unique
token-layout match from the runtime patch grid and model metadata: either
patch tokens only, or patch tokens plus an explicitly counted CLS/register
prefix. If the output token count cannot be explained uniquely, Phase 0 fails
closed. The report records the verified grid and prefix count; no fixed DINO
token count is assumed.

## Pair capture: raw patch features at early and late windows

~~~bash
MUJOCO_GL=osmesa python scripts/collect_patch_pairs.py \
  --matrix /path/to/libero_matrix.json \
  --checkpoint repos/CoRP/checkpoints/corp_libero.pt \
  --split-json docs/data/tta_split_libero_spatial.json \
  --phase0-report runs/patch_diag/phase0_interface.json \
  --suite libero_spatial --lighting-id 2292 \
  --roles source_fit,source_validation \
  --early-chunks 0,1 --late-chunks 6,7,8,9 \
  --seed 7777 --executed-k 12 \
  --out runs/patch_diag/source_patch_pairs.npz
~~~

The collector advances a source initial-state episode under the known target
lighting configuration using Frozen CoRP actions. At selected replan boundaries
it snapshots the shifted simulator state, restores that state in a Clean twin,
and checks model, physical state, controller, camera, policy-state, and patch
shape equivalence. It does not inspect task success or reward to select or stop
records; it runs to the requested chunk horizon or environment termination.
Failed pairs remain in the artifact for audit but are excluded downstream.

Outputs:

- source_patch_pairs.npz: per-view Clean/Shift pre-Resampler visual-token
  tensors, state inputs,
  state/language tokens, pooled language, explicit z0, and paired actions;
- source_patch_pairs.manifest.json: split role, state/chunk/step, pair checks,
  digests, verified patch grid/prefix-token layout, shapes, and file hashes.

The historical 37 early / 42 late records are a reference point, not a quota.
A different revision or collection run must report its actual count and
pairing-pass count.

## Phase 1: domain information versus content variability

~~~bash
python scripts/analyze_patch_pairs.py \
  --pairs runs/patch_diag/source_patch_pairs.npz \
  --out runs/patch_diag/phase1_report.json
~~~

Per view and window, the script reports:

- paired Clean/Shift channel-mean plus log-channel-std sensitivity (D_style);
  each state first averages its paired chunks, then states are weighted equally;
- Clean-Clean between-state variability at the same chunk index (D_content);
  cross-state pairs are averaged within chunk, then chunk indices are weighted equally;
- R_sep = D_style / D_content, using a source_fit-only state-level feature scale;
- source_fit Clean prototype-distance AUROC on source_validation states;
- logistic-regression AUROC trained on source_fit paired states and tested on
  disjoint source_validation states;
- state-cluster bootstrap confidence intervals for held-out AUROC;
- average N-by-N pre-Resampler visual-token self-similarity matrices, where N
  is the observed visual-encoder sequence length. If prefix tokens are present,
  they are included and identified separately in the report.

Self-similarity matrices are averaged within initial state first, then across
states equally.

The classifier is trained and tested at initial-state granularity: all
observations from a state stay in one split role. Early and late windows are
analyzed separately to avoid mixing manipulation phases.

**Interpretation:** high AUROC establishes that the known lighting-2292 signal
is separable across held-out source initial states. It does not establish
unknown-target generalization, causal responsibility, or action benefit. Low
AUROC does not prove that the Resampler or DiT is the bottleneck. R_sep is
descriptive, not a universal pass/fail law.

## Paired Patch Replacement: B0-B3

~~~bash
python scripts/paired_patch_replacement.py \
  --pairs runs/patch_diag/source_patch_pairs.npz \
  --phase1-report runs/patch_diag/phase1_report.json \
  --matrix /path/to/libero_matrix.json \
  --checkpoint repos/CoRP/checkpoints/corp_libero.pt \
  --out runs/patch_diag/b0_b3_report.json
~~~

Every branch uses the same source pair and explicit z0:

- **B0:** Clean patches plus Clean language/proprio context;
- **B1:** Shift patches plus Shift language/proprio context;
- **B2:** same-state Clean patches substituted into Shift context;
- **B3:** another state's Clean patches substituted into Shift context, matched
  by role, early/late window, and chunk index.

The report measures Resampler-token RMS distance and normalized/controller
action deltas (XYZ, SO(3) geodesic rotation, gripper, and clipping), with B0
as the reference. B0 is the Clean Frozen CoRP action under the same explicit
z0, not an expert action. Thus B2/B3 action "recovery" means consistency with
Clean Frozen behavior, not correctness or task-success improvement. Compare
B2 against both B1 and the random-content negative control B3.

**Causal interpretation guard:** B2 is a privileged offline mediator
intervention under a known target lighting configuration. Strong B2 recovery
and weaker B3 recovery support the claim that same-state visual patch content
mediates downstream action change for this known perturbation. They do not
establish unknown-target generalization, a deployable adaptation signal, or
closed-loop success benefit.

B0-B3 aggregates are state-balanced: chunks are averaged within each initial
state, then states are averaged equally. Do not treat multiple chunks from one
state as independent samples.

## Execution order and stop rules

1. Phase 0 must pass before pair collection.
2. Collect source-only paired patches; inspect the manifest and pairing failures.
3. Run Phase 1. Report every view/window independently.
4. Run B0-B3 on the same pairing-pass rows and explicit z0 values.
5. Consider Phase 2 visual alignment only if B2 restores Resampler/action
   outputs materially better than B1 and B3, while Clean identity remains
   acceptable. Any resulting method is still calibrated on a known perturbation
   and needs a separate fresh perturbation/task/suite for an unknown-domain claim.

No training, adapter update, OT, AdaIN, target-dev rollout, or target-test
access is part of these steps. These diagnostics consume the known target
lighting appearance on source initial states, so they must not be described as
fully source-free or unknown-target evaluation.

## Observation-freshness fix and re-run (2026-11-19)

Root cause of the previous 22/110 pairing pass rate was not `restore()`:
the Shift branch encoded the observation returned by the last `step()`, while
the Clean branch restored, called `sim.forward()`, and force-refreshed
observables. That stale-vs-fresh comparison is a different observation time.

The collector now keeps two tracks:

- execution uses the original returned Shift observation and the original
  frozen action chunk (rollout semantics preserved);
- captured pairs use a refreshed Shift observation after `sim.forward()` and
  force-updated observables, exactly matching the Clean twin refresh.

Every capture records `observation_freshness` (qpos/qvel/time deltas, raw and
normalized state deltas, state-token L2). `snapshot()` now also records MuJoCo
site definitions and dynamic site poses, and `policy_eef_pos` uses
`site_xpos[eef_site_id]` while `policy_eef_quat` uses the body quaternion,
matching the robot observables.

Freshness pilot (before re-collection):

| state | chunk | returned-vs-refreshed normalized state | state token L2 |
|---|---:|---:|---:|
| 0 | 0 | 0.000454 | 0.00420 |
| 0 | 1 | 0.010641 | 0.06299 |
| 4 | 0 | 0.000894 | 0.00614 |
| 4 | 1 | 0.011391 | 0.06978 |

This exactly matches the old pass/fail split at 1e-3.

After the fix, re-collected source pairs:

```text
records_total: 111
records_pairing_pass: 111
source_fit early 20/20, late 37/37
source_validation early 20/20, late 34/34
```

Fresh Phase 1 (state-balanced, source state splits):

| view/window | D_style | D_content | R_sep | held-out AUROC |
|---|---:|---:|---:|---:|
| agentview early | 1.610 | 1.622 | 0.993 | 1.000 |
| agentview late | 1.898 | 1.904 | 0.997 | 1.000 |
| eye-in-hand early | 2.391 | 2.227 | 1.073 | 1.000 |
| eye-in-hand late | 1.282 | 2.229 | 0.575 | 0.950 |

Fresh B0-B3 (same explicit z0, state-balanced):

| split | B1 action MAE | B2 action MAE | B3 action MAE | B2 recovery | B3 recovery |
|---|---:|---:|---:|---:|---:|
| fit early | 0.0522 | 0.0 | 0.0502 | 1.000 | 0.039 |
| fit late | 0.1246 | 0.0 | 0.3323 | 1.000 | -1.666 |
| val early | 0.0552 | 0.0 | 0.0494 | 1.000 | 0.105 |
| val late | 0.1355 | 0.0 | 0.3631 | 1.000 | -1.680 |

B2 is an exact same-state Clean-patch replacement and is privileged; its zero
action error is an implementation/mediation sanity check, not a deployable
method. B3 is the different-state negative control. No target-dev/test states,
success labels, rewards, or closed-loop outcomes are used.

## Closed-loop same-state Clean-patch oracle (ceiling check)

Purpose: determine whether restoring the correct same-state Clean visual
patches at every replan can change closed-loop task success, before designing
any deployable adapter.

Protocol: target-dev pilot states 36/37/43/45, Shift environment, Frozen CoRP
rollout. At each replan boundary the Shift simulator snapshot is restored into
a Clean twin with the same model body alignment and refreshed observables; the
Clean twin's pre-Resampler patches replace the Shift patches while Shift
language/proprio and the explicit z0 are retained. This is privileged and is
not a deployment method.

| state | Frozen | Same-state Clean-patch oracle |
|---|---:|---:|
| 36 | 1 | 1 |
| 37 | 1 | 1 |
| 43 | 0 | 1 |
| 45 | 0 | 1 |
| total | 2/4 | 4/4 |

The oracle preserves both Frozen successes and rescues both Frozen failures.
This establishes a positive closed-loop ceiling for visual-patch restoration
on this pilot. It does not establish an unlabeled deployable signal, target-test
generalization, or that the exact B2 token substitution is implementable.

### Full Gate-1 oracle (10 target-dev states x 3 seeds)

Target-dev pilot states 34/35/36/37/39/43/44/45/48/49, env/z0 seeds
7777/8888/9999, Frozen vs privileged same-state Clean-patch oracle.

| seed | Frozen | Oracle |
|---|---:|---:|
| 7777 | 2/10 | 7/10 |
| 8888 | 0/10 | 9/10 |
| 9999 | 4/10 | 8/10 |
| total | **6/30** | **24/30** |

Paired transitions across 30 episodes:

```text
fail -> success: 19
fail -> fail:     5
success -> success: 5
success -> fail:   1
```

The oracle strongly raises the closed-loop ceiling, but is not perfect:
six oracle failures remain and one Frozen success is broken (state 44,
seed 9999). This is a positive Gate 1, not proof of a deployable method.

### Gate 0 audit fixes and audited oracle pilot

Controller fix: `_controller_compare()` now computes numeric errors for
`goal_pos` and `goal_ori` instead of only checking shape/finiteness. New tests
cover goal-pos mismatch, goal-ori mismatch, and NaN. `policy_eef_quat` uses
quaternion sign-invariant angular distance; `eef_site_id` must be valid and
non-negative.

The oracle now refreshes the Shift derived state before snapshot, audits
model/state/camera/controller pairing at every oracle replan, and records
snapshot/model/controller digests, clean patch hash, z0 hash, action hash,
state-token L2 and freshness delta. The audited 4-state pilot:

| state | Frozen | audited oracle |
|---|---:|---:|
| 36 | 1 | 1 |
| 37 | 1 | 1 |
| 43 | 0 | 1 |
| 45 | 0 | 1 |
| total | 2/4 | 4/4 |

Every oracle replan reports `pairing_pass=true`. The earlier full 10-state x
3-seed oracle (6/30 -> 24/30) was collected before this audit and is not
claimed as audited; only the 4-state audited pilot is reported here.

### Oracle freshness control: Frozen-native vs Frozen-refreshed vs Oracle

The previous oracle used refreshed Shift context while Frozen used the
returned observation, so the comparison mixed visual replacement with
observation freshness. The control adds `frozen_refreshed` (same refreshed
Shift observation, Shift patches) and uses identical state/env/z0 rules.

| state | Frozen-native | Frozen-refreshed | Clean-patch oracle |
|---|---:|---:|---:|
| 36 | 1 | 1 | 1 |
| 37 | 1 | 0 | 1 |
| 43 | 0 | 0 | 1 |
| 45 | 0 | 0 | 1 |
| total | 2/4 | 1/4 | 4/4 |

Observation freshness alone does **not** produce the oracle gain: it changes
2/4 to 1/4 (loses state 37). The oracle recovers state 37 and rescues 43/45
under the same refreshed context. Thus the audited patch-only gain is not a
freshness artifact, though it is still a privileged oracle.

Every oracle replan passes context parity: state input, state token, language
tokens, and pooled language differences are 0.0; only patches differ. Both
Frozen and oracle branches now record patch/z0/state/action hashes.

### Gate 1A: source-prior domain score is not yet a usable gate

Tested the actual TP-TVA per-view score

D_v = state-balanced prior mean / log-std distance, calibrated on source_fit
Clean only; evaluated on disjoint source_validation states, early vs late.

| view/window | tau policy | fit Clean FPR | val Clean FPR | fit Shift act | val Shift act | val AUROC |
|---|---:|---:|---:|---:|---:|---:|
| agentview early | train max | 0.00 | 0.70 | 1.00 | 0.90 | 0.80 |
| agentview late | train max | 0.00 | 0.80 | 1.00 | 1.00 | 0.82 |
| eye-in-hand early | train max | 0.00 | 0.50 | 1.00 | 1.00 | 0.94 |
| eye-in-hand late | train max | 0.00 | 0.70 | 0.20 | 0.70 | 0.61 |

The threshold calibrated on 10 source_fit Clean states does not transfer:
held-out Clean FPR is 50-80%, and eye-in-hand late has weak Shift activation.
Therefore this mean/log-std domain score cannot currently implement the Clean
bypass or the TP-TVA gate. Gate 1A does **not** pass.

Implication: do not train the Adapter with this gate. Next candidate signals
must be calibrated on disjoint Clean states and evaluated by a direct
Clean-FPR/Shift-activation tradeoff; possible alternatives are cross-attention
energy consistency or differential-feature statistics, not another global
mean/std distance.
