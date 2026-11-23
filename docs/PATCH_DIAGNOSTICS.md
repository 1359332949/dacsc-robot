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

### Gate 1A-2: Resampler attention response also fails as a gate

Instrumentation was verified: manual attention-aware Resampler forward matched the
original Resampler to `5.2e-6` max token error; actions were still generated
with the original Resampler.

Candidate descriptors per view (attention entropy, effective support, max
probability, LogSumExp response) were calibrated on source_fit Clean states and
tested on disjoint source_validation states.

| view/window | val Clean FPR | val Shift activation | val AUROC |
|---|---:|---:|---:|
| agentview early | 0.40 | 0.40 | 0.39 |
| agentview late | 0.10 | 0.30 | 0.55 |
| eye-in-hand early | 0.10 | 0.50 | 0.70 |
| eye-in-hand late | 0.00 | 0.00 | 0.52 |

Action-deviation association was weak and inconsistent:

| view/window | Pearson(attention distance, action delta) |
|---|---:|
| agentview early | -0.23 |
| agentview late | -0.17 |
| eye-in-hand early | +0.26 |
| eye-in-hand late | -0.07 |

Neither the global mean/log-std gate nor the Resampler attention-response gate
separates held-out Clean from Shift reliably, and neither predicts action
deviation. Gate 1A-2 does not pass. Do not train the Adapter with either gate.

The correct next step is not to add differential features, cross-view cycles,
and temporal transport simultaneously; that would replace one unvalidated gate
with three. The research question must be re-examined at the signal level.

### Correction: Gate 1A prior-statistics bug and re-run

The first Gate 1A implementation mixed two different statistics: it used the
cross-state std of patch channel means as the reference for the within-sample
channel log-std. The corrected implementation estimates three separate
source-fit prior quantities:

- `prior_mu = mean_state(mu)`
- `prior_mu_scale = std_state(mu)`
- `prior_logsd = mean_state(log_std)`

and scores

D_v = mean((mu - prior_mu)^2 / (prior_mu_scale^2 + eps))
      + alpha * mean((log_std - prior_logsd)^2)

with a fixed alpha=1.0. Failed pairs are now rejected at the statistical
entry, and the threshold is reported as `tau_quantile95` rather than a 5% FPR
threshold (10 calibration states cannot guarantee 5% FPR).

Re-run on the same fresh NPZ:

| view/window | val Clean FPR | val Shift activation | val AUROC |
|---|---:|---:|---:|
| agentview early | 0.90 | 1.00 | 0.81 |
| agentview late | 0.80 | 1.00 | 0.79 |
| eye-in-hand early | 0.50 | 1.00 | 0.93 |
| eye-in-hand late | 0.70 | 0.70 | 0.61 |

The corrected statistics do not rescue the gate: held-out Clean FPR remains
50-90%. Gate 1A still does not pass.

### Gate 1A-2 attention parity and corrected statistics

The attention-aware Resampler matches the original on both outputs:

```text
token max error:     5.25e-6
attention max error: 4.68e-6
```

The attention gate still fails (held-out AUROC 0.39-0.70; eye-in-hand late
activation zero). State-level aggregation (20 state-role groups per window)
gives Pearson correlations in [-0.27, +0.25] with 95% bootstrap CIs crossing
zero for all view/windows. Attention descriptors are entropy, effective
support, max probability and LogSumExp; entropy/effective support are
monotone transforms, so they are not independent evidence.

Conclusion: neither the corrected global prior nor the audited Resampler
attention signal is a reliable policy-relevant gate. Do not train the Adapter.

### Gate 1A calibration audit and Gate 1A-3 measurement audit

Prior audit on the corrected Gate 1A:

- no channel hit the 1e-3 floor for any view/window;
- source-fit channel mean scales ranged 0.099-2.77 across views;
- 10 calibration states are still too few for a 5% distribution-free
  threshold; the reported threshold is therefore only `tau_quantile95` /
  `tau_crossfit`, never a guaranteed 5% FPR.

Cross-fitted threshold (prior on 9 fit states, threshold on the held-out fit
state) changes the operational picture:

| view/window | tau_crossfit_max val Clean FPR | val Shift activation | val AUROC |
|---|---:|---:|---:|
| agentview early | 0.0 | 0.0 | 0.81 |
| agentview late | 0.0 | 0.3 | 0.79 |
| eye-in-hand early | 0.1 | **1.0** | **0.93** |
| eye-in-hand late | 0.1 | 0.1 | 0.61 |

So the corrected conclusion is not "mean/log-std has no domain information".
It is:

- agentview scores do not activate on Shift under a clean-preserving threshold;
- eye-in-hand **early** has a candidate gate (10% Clean FPR, 100% Shift
  activation on 10 held-out states), but the calibration set is tiny;
- eye-in-hand late has no usable activation.

Gate 1A-3 fixed two measurement issues:

1. action deviation now uses **clipped controller actions** and the first
   executed K steps, split into XYZ/rotation/grip;
2. attention differences use per-query/head JS divergence, entropy change and
   max-probability change, not a mixed-dimension Euclidean distance.

State-level correlations (20 fit+val states; source-fit/validation separately
reported) reveal one strong candidate:

| view/window | metric -> action dim | Pearson | 95% CI | fit / val Pearson |
|---|---|---:|---:|---:|
| eye-in-hand late | JS -> rotation | **0.787** | [0.613, 0.906] | 0.914 / 0.771 |
| eye-in-hand late | JS -> XYZ | **0.690** | [0.489, 0.871] | 0.746 / 0.825 |
| eye-in-hand late | entropy change -> XYZ | **0.687** | [0.380, 0.847] | 0.863 / 0.684 |
| eye-in-hand late | max-prob change -> XYZ | **0.671** | [0.420, 0.880] | 0.793 / 0.717 |
| eye-in-hand early | JS -> XYZ | 0.506 | [0.179, 0.771] | 0.350 / 0.696 |

Agentview correlations are weak/inconsistent. This is a **policy-sensitivity
signal**, not a validated gate: it is view- and phase-specific, the calibration
set is small, and it has not been closed-loop tested. Do not train the Adapter
from this diagnostic alone.

### Gate 1A-4: deployable shift-only attention proxy

The strong eye-in-hand late correlations in Gate 1A-3 use paired Clean/Shift
JS/entropy changes, which are privileged and unavailable at deployment. A
deployable proxy was therefore tested: the Shift observation's attention
descriptor distance to the source-fit Clean prior, using no Clean patch.

Gate result (threshold from source-fit Clean only):

| view/window | val Clean FPR | val Shift activation | val AUROC |
|---|---:|---:|---:|
| agentview early | 0.2 | 0.0 | 0.27 |
| agentview late | 0.2 | 0.3 | 0.46 |
| eye-in-hand early | 0.1 | 0.3 | 0.55 |
| eye-in-hand late | 0.0 | 0.0 | 0.39 |

Correlation with executed-action deviation is weak/inconsistent; the strongest
magnitude is a negative relation for eye-in-hand early XYZ (r=-0.45, CI
[-0.68,-0.13]), not the privileged positive wrist-late JS relation.

Conclusion: the policy-sensitivity captured by paired attention changes is not
recovered by a simple shift-vs-source-prior descriptor. It is diagnostic
evidence, not a deployable gate. Gate 1A-4 does not pass; no Adapter training.

### Eye-in-hand early candidate gate vs executed action relevance

Validation-only state-level audit (10 source_validation states):

| action dimension | Pearson | 95% state bootstrap CI | Spearman |
|---|---:|---:|---:|
| norm | 0.469 | [-0.628, 0.875] | 0.236 |
| XYZ | 0.464 | [-0.631, 0.869] | 0.224 |
| rotation | 0.176 | [-0.836, 0.948] | 0.091 |
| gripper | -0.303 | [-0.727, 0.222] | -0.375 |

High-score vs low-score states have nearly identical XYZ action deviation
(0.1030 vs 0.1025). Thus the eye-in-hand early domain score is a candidate
lighting detector, but **not** demonstrated to be a task-risk detector. Do not
promote it to a gate on this evidence.

Gate 1A-4 and Gate 1A-3 now reject failed pairs and unexpected roles at the
statistical entry. Gate 1A-3 reruns identically on the all-pass data. The
cross-fitted Gate 1A threshold remains non-deployable: 10 source states are not
an independent calibration set, and the prior/threshold models are not nested
per validation score. Treat eye early as a hypothesis, not a pass.

### Offline decision audit: wrist-late confounding and source action spread

Wrist-late paired attention JS vs executed action deviation was decomposed
without any new training.

State-level raw vs within-state (chunk-centered) vs partial (state fixed
effects + clean action norm):

| action dim | state-level r | within-state r | partial r |
|---|---:|---:|---:|
| norm | 0.378 | -0.012 | -0.053 |
| XYZ | 0.690 | 0.246 | 0.341 |
| rotation | 0.787 | 0.214 | 0.221 |
| gripper | 0.242 | -0.059 | -0.123 |

The strong state-level wrist-late correlation is largely between-state. Within
a state, the association is much weaker (XYZ r~0.25; rotation r~0.21), so
wrist-late attention is not yet evidence for a deployable per-observation
predictor; it is at least partly a state-difficulty indicator.

Source-fit paired executed-action deviation distribution:

| window | dim | mean | std | p50 | max | ICC(state) |
|---|---|---:|---:|---:|---:|---:|
| early | XYZ | 0.097 | 0.025 | 0.094 | 0.147 | 0.683 |
| late | XYZ | 0.179 | 0.069 | 0.162 | 0.352 | 0.328 |
| early | rotation | 0.014 | 0.003 | 0.015 | 0.023 | 0.614 |
| late | rotation | 0.032 | 0.015 | 0.028 | 0.068 | 0.525 |
| late | gripper | 0.349 | 0.513 | 0.002 | 1.819 | 0.276 |

There is usable within-state action variation, especially late XYZ/gripper, so
a source-supervised predictor is not impossible. But a large fraction of the
signal is state-level (ICC 0.23-0.68), and any predictor must be validated on
new states, not the consumed 10 validation states. Gate 1A remains frozen; no
Adapter is trained.

### Correction: deterministic offline decision audit

The first offline audit reconstructed group indices in a different order from
the aligned arrays. The script is now index-safe: groups are built from
explicit aligned positions, input record order is irrelevant, all rows must
have unique `feature_row`, `pairing_pass=True`, and an allowed role. Added a
shuffle-invariance regression test and a failed-pair rejection test.

The corrected numbers are unchanged from the provisional report:

| action dim | state-level r | within-state r | partial r |
|---|---:|---:|---:|
| norm | 0.378 | -0.012 | -0.053 |
| XYZ | 0.690 | 0.246 | 0.341 |
| rotation | 0.787 | 0.214 | 0.221 |
| gripper | 0.242 | -0.059 | -0.123 |

`ICC(state)` is renamed to `between_state_variance_share`; it is a descriptive
variance share, not a mixed-model ICC estimate.

Source paired action spread now includes zero inflation and tails:

| window | dim | mean | p50 | p75 | p90 | max | P(>0.1) | between-state share |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| early | XYZ | 0.097 | 0.094 | 0.115 | 0.121 | 0.147 | 0.45 | 0.684 |
| late | XYZ | 0.179 | 0.162 | 0.206 | 0.292 | 0.352 | 0.865 | 0.328 |
| late | gripper | 0.349 | 0.002 | 0.808 | 1.194 | 1.819 | 0.378 | 0.276 |

The stored `controller_actions` are the clipped controller-space commands used
by the collector (first 12 steps); this is recorded in the audit output, but a
future collector should persist the executed action hash per step.

Even after correction, the wrist-late association is largely between-state.
No Adapter training.

### P1 completion and P2 photometric diagnosis

P1: offline audit now checks attention-report/manifest bidirectional uniqueness,
role/state/chunk consistency, NPZ row counts/shapes/finiteness/index bounds,
and at least 12 action steps. The measured quantity is renamed
`delta_controller_command` (clipped controller command, not post-controller
physical motion).

P2 collected paired raw images at source states, both views, early/late. The
dominant photometric change of Lighting 2292 is global darkening plus contrast
loss:

| view/window | mean lum delta | lum std delta | contrast delta | shadow frac delta | edge mean delta |
|---|---:|---:|---:|---:|---:|
| agentview early | -0.071 | -0.035 | -0.092 | +0.0058 | -0.0018 |
| agentview late | -0.072 | -0.039 | -0.097 | +0.0039 | -0.0031 |
| eye-in-hand early | -0.044 | -0.027 | -0.058 | -0.0009 | -0.0017 |
| eye-in-hand late | -0.101 | -0.047 | -0.134 | +0.0073 | -0.0026 |

RGB mean vector shift norm is 0.12 (agentview) and 0.08-0.18 (eye-in-hand);
saturation change is small. Local luminance-contrast variation also drops in
late wrist (local std max delta -0.06), and edge density declines, so this is
not a pure global offset. A single fixed brightness/contrast transform is a
plausible but not guaranteed candidate; P2 does not select transform parameters
and does not use success rates.

No Adapter, no target-test. Next: freeze one input transform and run a paired
Clean/Shift pilot only after the transform and evaluation thresholds are
pre-registered.

### P3: fixed photometric-transform pilot (negative probe)

A single fixed per-view affine luminance transform was fit only on source-fit
Clean/Shift image statistics and frozen before the pilot:

- agentview: a=1.238, b=-0.0215
- eye-in-hand: a=1.160, b=+0.0063

It was applied to both Clean and Shift images; same state/env/z0/controller.

| domain | Frozen | Transform |
|---|---:|---:|
| Clean | 4/4 | **3/4** |
| Shift | 2/4 | **1/4** |

The transform broke Clean state 37 and did not rescue Shift 43/45. Therefore a
simple global brightness/contrast correction is not sufficient for this
Lighting-2292 setting. This is a probe result, not a method; no further affine
parameter tuning was performed. Do not train an Adapter on this signal.

### P1/P2/P3 audit corrections

P1:
- feature_row must be an integer in [0, N) on both sides; negative indexing is rejected;
- tests cover negative, duplicate, and out-of-range rows.

P2:
- `edge_mean` renamed `mean_gradient_magnitude`; it is not edge density and scales
  with global contrast;
- duplicate `local_edge_mean_mean` removed from the reported feature list;
- analysis enforces fit/validation role purity, disjoint state IDs, unique
  (role,state,chunk,view) and reports state-level paired bootstrap CI.

P3:
- domains/modes/executed_k/max_steps/action-horizon validation added;
- transform and action clipping fractions recorded;
- smoke state 36: Clean transform agentview clipping ~6.1%, eye ~1.9%;
  Shift transform agentview ~6.0%, eye ~0.01%.
- existing full P3 result is unchanged: Clean 4/4 -> 3/4, Shift 2/4 -> 1/4.

Renaming/clipping did not change the P3 conclusion. No Adapter, no target-test.

### Failure-stage instrumentation (P3 follow-up)

Instrumented trajectories for state 37 clean/shift (frozen vs transform),
state 43/45 shift (frozen vs transform), and state 36 clean frozen reference.
Per step recorded raw/clipped commands, post-controller `sim.data.ctrl`, EEF
pose, gripper qpos, object body poses and contact count.

Paired frozen-vs-transform first divergence:

| case | frozen | transform | first EEF >1mm | first action >0.1 | first gripper >0.1 | frozen/transform close command |
|---|---|---:|---:|---:|---:|---:|
| 37 clean | success | fail | 3 | 13 | 49 | 50 / 49 |
| 37 shift | success | fail | 4 | 25 | 49 | 55 / 49 |
| 43 shift | fail | fail | 3 | 29 | 47 | **never / 47** |
| 45 shift | fail | fail | 2 | 2 | 43 | 43 / 49 |

Key observations:

- state 37 transform diverges in EEF at steps 3-4, long before gripper close;
  the affine transform changes the approach trajectory, not just grasp timing;
- state 43 Frozen never sends a close command; transform sends close at step
  47 but still fails, so closing alone is insufficient;
- state 45 Frozen and transform both close, but at different steps (43 vs 49)
  and both fail; this is a late contact/grasp-phase failure.

The failure is state-dependent: early approach divergence (37), missing close
decision (43), and late grasp/contact timing (45). A single global input
transform cannot address all three. No Adapter is trained.

### Corrected contact attribution (target-specific)

The earlier instrumentation used global `ncon > 0`; that is always nonzero due
table/robot contacts. The corrected instrumentation resolves the BDDL goal
object (`akita_black_bowl_1`), its body/geoms, and finger geoms, and records
target-finger contact separately.

| case | success | close cmd | first target-finger contact | target lift | EEF-target dist at close |
|---|---:|---:|---:|---:|---:|
| 36 clean frozen (ref) | 1 | 45 | 55 | 0.108 | 0.075 |
| 37 clean frozen | 1 | 50 | 60 | 0.083 | 0.088 |
| 37 clean transform | 0 | 49 | 59 | **0.027** | 0.074 |
| 37 shift frozen | 1 | 55 | 59 | 0.052 | 0.086 |
| 37 shift transform | 0 | 49 | **none** | 0.000 | **0.151** |
| 43 shift frozen | 0 | **none** | none | 0.000 | — |
| 43 shift transform | 0 | 47 | none | 0.000 | 0.156 |
| 45 shift frozen | 0 | 43 | none | 0.000 | **0.162** |
| 45 shift transform | 0 | 49 | none | 0.000 | 0.127 |

Correction: no forced-close intervention was run. The earlier phrase
"forced close still failed" was incorrect. State 43 only shows that the
policy's own close command is absent under Frozen, while the affine transform
emits close at step 47 but at 0.156 m from the target with no target contact.

State 45 is not a late-contact failure: both modes close at 0.13-0.16 m from
the target and never touch it. State 37 clean transform does touch the target
but lifts only 0.027 m vs 0.083 m for Frozen; shift transform closes at
0.151 m with no contact. The common pattern is **approach/close position**,
not just gripper timing.

### Experiment A: State 37 fixed-physical-state three-input trace

Frozen state-37 Clean rollout, first 24 control steps. At every step the same
physical state was re-rendered under Shift lighting and four encodings were
compared with the same language/state/z0: Clean, Clean+affine Transform, and
Shift. Shift/Clean camera parity was 24/24.

Target geometry audit:

- BDDL target `akita_black_bowl_1` maps uniquely to body 23
  `akita_black_bowl_1_main`;
- target has 41 geoms; finger set has 6 geoms;
- EEF-to-target distances below use the target body origin, which is not
  necessarily the grasp point; this is a limitation, not a grasp-pose metric;
- no target-finger contact occurred in the first 24 steps.

Mean over 24 steps:

| comparison | DINO rel L2 | DINO cosine | Resampler rel L2 | action XYZ pre | action XYZ post | rotation pre | gripper pre/post |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean vs Clean+Transform | 0.268 | 0.961 | 0.179 | 0.035 | 0.035 | 0.0105 | 0.0006 / 0.0 |
| Clean vs Shift 2292 | 0.797 | 0.667 | 0.441 | 0.243 | 0.243 | 0.0270 | 0.0053 / 0.0 |

Observations:

- Shift produces a much larger DINO/Resampler/action change than the affine
  transform;
- the Resampler attenuates both perturbations (0.268->0.179 relative for
  Transform; 0.797->0.441 for Shift);
- the affine transform already changes the same-observation action (XYZ
  ~0.035), but the Shift action change is ~7x larger;
- gripper post-clip difference is exactly zero in this window, so early action
  differences are dominated by XYZ/rotation, not gripper clipping;
- `theta_rel` from EEF quaternion to target body quaternion is ~1.57->1.44 rad;
  this is body-origin orientation, not a validated grasp-point orientation.

This supports a forward causal path visual change -> representation change ->
action change at the same physical state. It does not yet show that this causes
the closed-loop failure across states; one state is not enough. No Adapter.

### Experiment A extension: four-state fixed-observation trace

Four target-dev states (36, 37, 43, 45), 20 control steps each, same physical
state, same z0/language/state, three inputs: Clean, Clean+Transform, Shift.
All 20/20 state/camera parity checks passed per state. No target-finger contact
occurred in these first 20 steps; EEF-target body-origin distance was still
0.37->0.27 m, so all traces are in the approach phase.

Mean over 20 steps:

| state | branch | DINO rel | Resampler rel | action XYZ | rotation | gripper pre/post |
|---|---|---:|---:|---:|---:|---:|
| 36 | Transform | 0.277 | 0.181 | 0.036 | 0.0072 | 0.0011/0.0002 |
| 36 | Shift | 0.754 | 0.421 | 0.290 | 0.0226 | 0.0039/0.0006 |
| 37 | Transform | 0.269 | 0.172 | 0.035 | 0.0098 | 0.0006/0.0 |
| 37 | Shift | 0.777 | 0.424 | 0.251 | 0.0294 | 0.0062/0.0 |
| 43 | Transform | 0.242 | 0.168 | 0.020 | 0.0055 | 0.0004/0.0 |
| 43 | Shift | 0.727 | 0.370 | 0.303 | 0.0160 | 0.0041/0.0004 |
| 45 | Transform | 0.267 | 0.156 | 0.030 | 0.0050 | 0.0008/0.0001 |
| 45 | Shift | 0.709 | 0.410 | 0.270 | 0.0213 | 0.0020/0.0 |

The ordering is consistent across all four states: Shift DINO ~0.71-0.78,
Resampler ~0.37-0.42, action XYZ ~0.25-0.30; Transform DINO ~0.24-0.28,
Resampler ~0.16-0.18, action XYZ ~0.02-0.04. The Resampler attenuates both
perturbations but does not eliminate them. Gripper post-clip differences are
near zero; early action differences are XYZ/rotation.

Closed-loop alignment (Frozen vs Transform trajectories):

| case | first EEF >1mm | first action >0.01 | first action >0.05 | first action >0.1 |
|---|---:|---:|---:|---:|
| 37 clean | 3 | 1 | 9 | 13 |
| 37 shift | 4 | 1 | 11 | 25 |
| 43 shift | 3 | 1 | 1 | 29 |
| 45 shift | 2 | 1 | 1 | 2 |

Action differences at the 0.01 scale precede EEF divergence in all four
paired cases, but the 0.01-scale difference is small; the larger 0.1-scale
divergence appears later (or, for state 45, at step 2). This is consistent with
a visual->action->approach chain, not proof that the small early action change
causes the eventual failure. Target geometry still uses the body origin, not a
validated grasp point; no Adapter training.

### Causal action intervention on State 37

Injected the one-step Transform action delta into the Frozen trajectory at the
first executed action, then continued Frozen policy.

| domain | intervention | success | injected max | first EEF >1mm | final EEF separation |
|---|---|---:|---:|---:|---:|
| clean | baseline | 1 | — | — | — |
| clean | full one-step | 1 | 0.048 | 25 | 0.046 |
| clean | XYZ-only one-step | 1 | 0.048 | 19 | 0.026 |
| shift | baseline | 1 | — | — | — |
| shift | full one-step | 0 | 0.019 | 19 | 0.084 |
| shift | XYZ-only one-step | 1 | 0.019 | 21 | 0.012 |

A single injected action difference does **not** reproduce the early step 3-4
EEF divergence seen under continuous Transform. In this one-state/one-seed
probe, continuous transformed observations over replans are required for the
early divergence. The full-delta Shift injection eventually failed while the
XYZ-only injection succeeded, but this is not yet replicated and the EEF
divergence appears late (step 19), not early. Do not treat this as causal proof
or a method result. No Adapter.

### P0-A/P0-B continuous intervention pilot

States 37/43/45, clean baseline vs shift baseline vs injecting the same-observation
Shift-Clean action delta into Clean Frozen. All interventions were computed at
each replan from the current Clean observation and a synchronized Shift rendering;
`p0b_chunk1/2` inject for the first 1/2 chunks, `fixed_chunk2` reuses the first
chunk's delta for 2 chunks.

| state | Clean baseline | Shift baseline | P0-B 1 chunk | P0-B 2 chunks | fixed 2 chunks |
|---|---:|---:|---:|---:|---:|
| 37 | success 142 | success 114 | success 103 | success 97 | success 139 |
| 43 | success 95 | **fail 220** | success 94 | success 94 | success 106 |
| 45 | success 91 | **fail 220** | success 93 | **fail 220** | success 89 |

Injection magnitudes were large (max 0.14-0.24), yet most Clean+injection runs
succeeded. The one failure (45, 2-chunk P0-B) is a single seed/episode and is
not replicated. This is evidence against the strong hypothesis that an
action-delta sequence alone is sufficient to reproduce the Shift failures;
continuous shifted observations / closed-loop state effects remain candidates.
No multi-seed confirmation yet.

### Bowl collision geometry audit

Target `akita_black_bowl_1` maps uniquely to body 23. It has 41 geoms: 40
collision geoms of MuJoCo type 6 with `dataid=-1` (primitive boxes) and one
visual geom. There is no mesh vertex data, so a rim-circle fit is not supported
by the collision representation. Finger bodies are left/right finger plus tip
bodies; raw closing-axis vector is ~[-0.04,-0.18,-0.98].

Consequence: the earlier "bowl rim circle" assumption is not established. The
next geometry metric must use the actual primitive collision geometries and
finger pads, not a fitted mesh rim. The body origin remains unsuitable as a
grasp point.

### Phase-swap pilot and exact OBB geometry

Fixed switch at step 24 (2 chunks), same state/env seed, synchronized at switch
with state/camera parity passed:

| state | Shift -> Clean | Clean -> Shift |
|---|---:|---:|
| 37 | success 105 | success 101 |
| 43 | success 96 | success 105 |
| 45 | success 113 | **fail 220** |

This is not a clean phase-effect result yet: state 43's earlier Shift baseline
failed, but here both phase-swap directions succeed. That indicates a
single-seed instability in the 43 Shift failure (or a setup difference); it
must be replicated before attributing 43 to early vs late phase. State 45 is
consistent with a late-Shift sensitivity, but one seed is not enough.

OBB geometry audit (initial settled state; 40 boxes, two pad collision geoms):

| state | left pad min distance | right pad min distance | closing-to-normal angle (L/R) |
|---|---:|---:|---:|
| 37 | 0.321 | 0.317 | 81.1 / 87.1 deg |
| 43 | 0.300 | 0.296 | 80.5 / 87.1 deg |
| 45 | 0.313 | 0.310 | 81.5 / 86.4 deg |

These are initial-state distances, not close-time distances. The exact
point-to-OBB tool is now available for replay instrumentation; close-time OBB
must be recorded in the next rollout. No Adapter, target-test untouched.

### Multi-seed phase-swap + close-time OBB

State 43/45, seeds 7777/8888/9999, switch step 24, same-state/camera parity.
Close OBB is measured on the first close command using exact point-to-OBB
distances between the two pad collision geoms and 40 bowl boxes.

| state | condition | success | close pad distances (representative) |
|---|---|---:|---|
| 43 | Clean baseline | 3/3 | 0.03-0.05 m |
| 43 | Shift baseline | **0/3** | 0.185-0.264 m when close occurred; 2/3 no close |
| 43 | Shift -> Clean | 3/3 | 0.02-0.04 m |
| 43 | Clean -> Shift | **2/3** | 0.02-0.05 m |
| 45 | Clean baseline | 3/3 | 0.02-0.04 m |
| 45 | Shift baseline | **0/3** | 0.06-0.13 m |
| 45 | Shift -> Clean | 3/3 | 0.02-0.04 m |
| 45 | Clean -> Shift | **2/3** | 0.02-0.04 m |

Corrected conclusion: the failure is not consistently attributable to early
or late Shift alone. Early Shift + late Clean always recovered; late Shift
alone usually recovered. Sustained Shift across both phases failed 6/6. The
close-time OBB distances show the Shift baseline is far from the bowl at the
close command, while successful phase-swap runs get within ~0.02-0.05 m.
This supports an approach/close-position failure under sustained shifted
observations, not a simple "late Shift" or "early Shift" mechanism.

The single clean->shift failures (one seed in each state) need more seeds; the
full-Shift failure is currently the robust effect (6/6 across both states).

### True 2x2 checkpoint fork (final micro-diagnostic)

Two independent prefix snapshots per seed: Clean step24 and Shift step24. `C->C`
and `C->S` share the Clean snapshot; `S->C` and `S->S` share the Shift snapshot.
Switch point is a replan boundary (24 = 2 x 12).

| state | prefix -> suffix | success |
|---|---|---:|
| 43 | C -> C | 3/3 |
| 43 | C -> S | 3/3 |
| 43 | S -> C | 3/3 |
| 43 | S -> S | 0/3 |
| 45 | C -> C | 3/3 |
| 45 | C -> S | 2/3 |
| 45 | S -> C | 2/3 |
| 45 | S -> S | 0/3 |

Across states: C->C 6/6, C->S 5/6, S->C 5/6, S->S 0/6.

This closes the phase question: after a Clean prefix, switching to Shift
usually still succeeds; after a Shift prefix, continuing Shift always fails.
The key variable is the **prefix physical state / early Shift exposure**, not
late Shift alone. State 45 has one seed where all suffix branches fail, so the
S-prefix effect is not perfectly deterministic; S->S remains robustly failed.

Per the research-scope decision, stop expanding this micro-diagnostic. The next
gate is P1 attention-mask audit, then a minimal target_dev-wide prototype; no
Adapter training before that.

### Target-dev baseline and P1 attention audit; first prototype negative

Full target_dev (10 states) Frozen:
- Clean: 8/10 success (34,35,36,37,43,44,45,49; failures 39,48)
- Shift 2292: 2/10 success (36,37 only)

P1 attention-mask audit, eye-in-hand, 20 steps, states 43/45:

| state | target mass clean/shift | finger mass clean/shift | background mass clean/shift | top64 Jaccard |
|---|---|---|---|---|
| 43 | 0.084 / 0.102 | 0.032 / 0.027 | 0.884 / 0.871 | 0.607 |
| 45 | 0.088 / 0.109 | 0.035 / 0.028 | 0.876 / 0.863 | 0.603 |

The attention is dominated by background; target/finger mass is low. Under the
audit's projected mask this means attention cannot be trusted as a foreground
mask for adaptation; if confirmed on more states, adaptation must be at the
DINO shallow/input side rather than DINO->Resampler attention weighting.

First minimal prototype: source-anchored pre-Resampler patch-token
normalization (per-view source mean/std, per-image target mean/std). It was
stopped after Clean 0/8: the global channel normalization destroys Clean
retention immediately. This prototype is rejected; no Shift run needed. The
negative result reinforces that a Clean-preserving constraint is mandatory and
simple global feature normalization is not acceptable.

No Adapter training; target-test untouched.

### Prototype 2: local patch-energy calibration

Source clean patch L2 norm is used as a per-view anchor; each target patch
vector is scaled by 1 + 0.2*tanh(log(anchor_norm / patch_norm)), with a hard
20% cap. No global channel mean/std, no attention mask, no training.

Clean gate (8 states that Frozen Clean solves):

| condition | success |
|---|---:|
| Clean Frozen baseline | 8/8 |
| Clean + local patch energy | **8/8** |

Shift gate (all 10 target_dev states, 2292):

| condition | success | states |
|---|---:|---|
| Shift Frozen baseline | **2/10** | 36,37 |
| Shift + local patch energy | **4/10** | 36,37,39,49 |

This is the first prototype that passes the Clean gate and improves Shift
without any target labels. It is still modest and not a method-level result:
43/45/44/48 remain failures, and only one epsilon was tested. Next: small
epsilon robustness check and failure-mode analysis on the new successes
(39/49) before any training.

### Epsilon sweep and multi-seed check: prototype 2 not robust

Clean/Shift target_dev, single seed 7777:

| eps | Clean | Shift |
|---:|---:|---:|
| 0.1 | 8/10 | 2/10 |
| 0.2 | 8/8 (8-state gate) | 4/10 |
| 0.3 | 8/10 | 1/10 |

The 0.2 result was seed-specific. Shift multi-seed at eps=0.2:

| seed | Shift success |
|---:|---:|
| 7777 | 4/10 (36,37,39,49) |
| 9999 | 4/10 (35,37,39,44) |
| 8888 | 0/9 observed before hang |

Clean was only run for seed 7777; multi-seed Clean retention is not yet
established. Conclusion: local patch-energy calibration is Clean-safe in the
single-seed gate, but its Shift gain is not stable across seeds. It is not a
method-level result and must not be treated as a validated prototype. No
Adapter training; target-test untouched.

### LPE protocol fix and decisive offline action-error test

Protocol fixes:
- LPE source anchor now reads only `source_fit` manifest rows and records NPZ/manifest SHA256;
- records include env_seed, z0_seed, commit, `max_scale_delta` and explicit
  `scale_range = [1-delta, 1+delta]`;
- per-episode flush added (a hung episode no longer loses completed records).

Decisive offline test on 57 source_fit paired records: compute Clean, Shift,
and Shift+LPE actions from the same stored patches/tokens/z0 and compare to
Clean action.

| metric | Shift | Shift+LPE |
|---|---:|---:|
| mean action L2 vs Clean | 0.16214 | **0.16208** |
| mean XYZ vs Clean | 0.14129 | **0.14146** |

LPE reduces the action deviation by ~0.03% in L2, and slightly worsens XYZ.
This means LPE does not act through the intended mechanism: its occasional
seed-specific success was not from reducing Shift-Clean action error.

Decision: stop LPE. Do not run the 3-seed matrix and do not stack temporal
residual on it. Return to representation-level questions; no Adapter training.

### r=6 rolling-horizon baseline and Oracle-gradient capacity probe

r=6 Frozen target_dev (same 10 states, clean/shift):

| horizon | Clean | Shift |
|---|---:|---:|
| r=12 (original) | 8/10 | 2/10 |
| r=6 (rolling) | 8/10 | **1/10** |

r=6 does not help: Shift success drops, and Clean state 43 regresses (state 39
is rescued). Therefore shortening replan interval is not a free improvement;
do not adopt r=6 solely to create overlap.

Oracle-gradient capacity probe at the pre-Resampler token interface (proxy for
a small LayerNorm at DINO->Resampler, because raw images are not stored in the
patch-pair NPZ). Zero-init residual LayerNorm, trained only on source_fit
Lighting pairs:

| metric | before | after |
|---|---:|---:|
| fit Oracle loss | 0.04815 | 0.04354 |
| held-out Oracle loss | 0.06261 | 0.06169 |
| held-out Clean action drift | — | 0.000538 |

Fit improves ~9.6%, held-out only ~1.5%, with non-zero Clean drift. This proxy
does not establish useful held-out capacity for a DINO LayerNorm; the exact
DINO-image path still needs raw images. Do not train DINO params from this.
Background/noise domains exist in the matrix but have not been paired/collected
for target_dev yet.

### P0 baseline ledger and P1 action metric

Formal paired baseline ledger (screening seeds 7777/8888/9999, 10 target_dev
states, fp32, sample_z, executed_k=12, same z0 schedule per seed):

| domain | seed 7777 | seed 8888 | seed 9999 | total |
|---|---:|---:|---:|---:|
| Clean Frozen | 7/10 | 6/10 | 8/10 | **21/30** |
| Shift Lighting 2292 | 5/10 | 3/10 | 3/10 | **11/30** |

The old Clean 8/10 / Shift 2/10 and r=6 8/10 / 1/10 are retained only as
legacy exploratory diagnostics and are not mixed into this ledger. The 10
initial states are the physical cluster; seeds are repeats on the same states.

Access-ledger fact: target-test status is `admission_accessed` (Lighting 2312
Frozen admission, seed 2014), not locked_unused. The split file still records
`locked_unused`; the access ledger supersedes it. This test is exploratory
admission only, not confirmatory.

P1 action metric module added: execution clip, XYZ error, SO(3) geodesic for
controller rotation-vector commands, separate gripper error, and state-balanced
aggregation. Tests cover identity, clip masking, rotation geodesic, and
all-dimension clipping. No proxy retraining was run.

### P1.1 controller scaling parity

`dacsc/action_metrics.py` now maps executed OSC_POSE controller commands to
physical units using the actual robosuite config:

- input [-1,1] -> XYZ output [-0.05,0.05] m;
- input [-1,1] -> rotation axis-angle output [-0.5,0.5] rad;
- gripper unchanged.

Parity check against the loaded LIBERO controller: max absolute difference
2.78e-17 on a random clipped action. Unit tests now cover physical scaling,
clip masking, SO(3) geodesic, identity, and all-dimension clipping (5/5 OK).

Provenance overlay added for `perturb_dev_clean_v2.json`: its captured
`target_test_status=locked_unused` predates the 2312 admission event; the
global access ledger (`admission_accessed`) supersedes it. Rollout records are
unchanged. P2 real RGB DINO collection remains the next step; no Adapter.

### P2 real RGB DINO LayerNorm oracle feasibility

Collected 20 real paired RGB observations (10 source_fit, 10 source_validation),
one fresh Clean/Shift rendering per physical state; pairing checks pass.

Predeclared theta: DINO block 0 norm1 only (768 parameters: 384 weight + 384
bias). Everything else in DINO, Resampler and DiT frozen. Oracle reference is
Clean Frozen theta0, fixed; action loss is unnormalized action MSE on z0.

| metric | init | final |
|---|---:|---:|
| fit Oracle loss | 0.004695 | 0.004636 |
| held-out Oracle loss | 0.005370 | 0.005357 |
| held-out Clean drift | — | 1.71e-7 |
| nonzero grad | — | yes (~0.003) |

Under this predeclared single LayerNorm group, 20 steps, lr=1e-4, held-out
improvement is ~0.24%, with negligible Clean drift. This is not evidence of
useful Oracle capacity; it only rejects this single parameterization/setting.
No second parameter group was selected on validation. P2 does not proceed to
deployment TTA from this result.

### A-closeout physical metrics and B0 velocity-head rank

P2 physical action metrics (real RGB, block0.norm1, 20 steps, lr=1e-4):

| metric | Shift vs Clean ref | Clean drift |
|---|---:|---:|
| normalized XYZ | 0.18134 | 0.000898 |
| physical XYZ | 0.009067 m | 0.0000449 m |
| normalized rotation | 0.023267 | 0.000220 |
| physical SO(3) | 0.011634 rad | 0.000110 rad |
| gripper | 0.000240 | 0.00000086 |

So the raw action gap is 9.1 mm / 0.0116 rad; the block0.norm1 Oracle update
only reduced raw loss from 0.005370 to 0.005357 at held-out, with negligible
but nonzero Clean drift.

B0 velocity-head audit on the actual checkpoint:

```text
velocity_projection.weight shape = [7, 768]
rank = 7 (full row rank)
sigma = [0.5149, 0.4287, 0.4194, 0.4070, 0.3671, 0.3541, 0.3431]
condition number = 1.50
```

Therefore an unconstrained hidden delta can realize any 7D velocity residual
through this linear head; an unbounded velocity-oracle is not evidence of
visual repair capability. B1 must impose ||delta h|| <= rho ||h||, and B2 must
restrict delta h to a sourced visual readout (FAVR-like), with flow integration
and physical action evaluation.

### B1/B2 bounded velocity oracle

Fixed flow step tau=0.5, same z0 action input, same Clean/Shift state and
Resampler path. W = actual 7x768 velocity_projection; rank 7, cond 1.50.

B1 (unconstrained minimum-norm hidden delta):
- 20/20 records satisfy ||delta h|| <= 0.1 ||h||;
- mean ratio = 0.0373.

This is expected from the full-row-rank 7D head and does not establish visual
repair capability: any 7D velocity residual is representable.

B2 (single bounded scalar along the Clean-Shift visual-token difference,
projected through W, alpha bounded by 1):
- mean relative residual = 0.601;
- mean alpha = -0.542.

So the simple visual direction leaves ~60% of the velocity residual
unexplained. B2 fails as a structured velocity oracle. No flow-integration
success/failure claim yet; no Adapter training.

### B-Flow: exact Euler trajectory, per-token B1, Oracle integration parity

Real sampler: N=10 Euler steps, x0=z0, dt=1/10. At each real flow state we
evaluated Clean and Shift velocity on the same x, not on z0 at tau=0.5.

B1 per-token reachability (rho=0.1 predeclared):

| metric | value |
|---|---:|
| mean token ratio | 0.0397 |
| mean P95 token ratio | 0.0805 |
| mean max token ratio | 0.1118 |
| states with max token ratio <= 0.1 | 3/20 |

So the mean/P95 token correction is within 10%, but the worst token exceeds
the predeclared bound in 17/20 pairs. Do not relabel rho post hoc.

Step 4 Oracle integration parity: adding W*W^dagger*(v_C-v_S) to the Shift
velocity at every real Euler step reproduces the Clean final action to
max |diff| = 2.38e-7 (float32). Corrected physical XYZ error = 2.2e-9 m,
versus uncorrected Shift error = 0.00873 m.

This proves the velocity interface and integration parity, but only under a
privileged unbounded hidden correction; it does not provide a deployable
visual readout. The strict per-token 0.1 bound is not met for all tokens.

### Fresh RGB re-collection and B-Flow-2

Freshness smoke on states 0/1 showed the old Clean images/state8 differed from
the unified-refresh path; Shift hashes matched. Full 20-pair set was therefore
re-collected with both sides using sim.forward -> _update_observables ->
_get_observations.

P2 on fresh RGB remains weak: held-out loss 0.005326 -> 0.005313 (~0.245%),
clean drift 1.69e-7; physical shift gap 8.998 mm / 0.01178 rad.

B-Flow-2 on fresh RGB, true 10-step Euler, per-token trace and bounded
correction:

| rho | physical XYZ | SO(3) | gripper |
|---:|---:|---:|---:|
| 0 Shift | 8.642 mm | 0.011883 rad | 2.38e-4 |
| 0.05 | 1.965 mm | 0.002872 rad | 3.80e-5 |
| 0.10 | 0.151 mm | 0.000237 rad | 1.02e-6 |
| 0.15 | 4.66e-6 m | 8.58e-6 rad | 9.9e-10 |
| 0.20 | 1.92e-9 m | 3.59e-9 rad | 9.9e-10 |

Per-token ratio: mean 0.0397, P95 0.0864, max 0.1631. By flow step it rises
from ~0.024 at step 2 to ~0.072 at step 9; by action position it is U-shaped
(~0.047 at ends, ~0.033 in the middle). Thus the strict per-token max does not
mean bounded correction is unusable: truncating at rho=0.10 still reduces
physical XYZ error from 8.64 mm to 0.151 mm in the 10-step integrated action.

This is a privileged Clean/Shift velocity oracle, not a deployable readout.
No Adapter, no target-test.

### B-Flow-3: target-dev first-decision headroom

30 target-dev pairs (10 states x 3 P0 seeds). The P0 CUDA z0 schedule was
regenerated and its SHA256 matched the ledger for all records; Clean/Shift
state8 max absolute difference was 0.0; pairing all passed. B-Flow-3 uses
z0_schedule[0] and the same 10-step Euler bounded oracle.

Bounded physical controller-command error at rho=0.10:

| group | n | Shift baseline XYZ | rho=0.10 XYZ |
|---|---:|---:|---:|
| Clean success / Shift fail | 13 | 5.94 mm | **0.020 mm** |
| both success | 8 | 4.71 mm | 0.122 mm |
| both fail | 6 | 5.25 mm | 0.025 mm |
| Clean fail / Shift success | 3 | 6.57 mm | 0.123 mm |
| all | 30 | — | mean 0.0586 mm, median ~0, P90 0.139 mm, max 0.519 mm |

The first-decision action chunk has ample bounded hidden headroom, including
the 13 core Clean-success/Shift-fail episodes. This is still a privileged
Clean/Shift Oracle; it does not provide a deployable readout. No Adapter, no
target-test.

### R0 minimal source readout: first attempt fails validation

Dataset: 4 z0 trajectories per source state, 10 Euler steps, 12 action tokens;
deployable features are Shift Resampler mean, action hidden h, state8, tau, k.
Teacher is r* = v_C - v_S at the uncorrected Shift trajectory state. Clean
samples use zero residual (No-Op).

Ridge R0 (h + state8 + tau + k; Clean No-Op weight 1; alpha selected by
state-grouped CV on source_fit only):

| metric | value |
|---|---:|
| source_fit velocity relative error | 0.284 |
| source_validation velocity relative error | **0.710** |
| Clean predicted-correction norm (val) | 0.175 |

The linear readout overfits the fit states and fails to generalize to held-out
source states; the Clean No-Op constraint is also not satisfied. This is a
negative R0 result: a simple source-trained single-frame linear readout does
not explain the Clean/Shift velocity residual across states. Do not add an
Adapter or deploy; first analyze state-conditioned label variance and feature
generalization.

### R0-2 feature ablations and identifiability

R0 dataset: source 20 states, 4 z0 trajectories, 10 flow steps, 12 action
positions; ridge with state-grouped CV, per-fold normalization (B0/B1/B2),
Clean No-Op samples, intercept unpenalized.

| feature set | CV relative | val relative | val aggregate | val cosine | Clean residual norm |
|---|---:|---:|---:|---:|---:|
| B0 h + state8 + tau/k | 0.621 | 0.675 | 0.684 | 0.764 | 0.124 |
| B1 visual mean + state8 + tau/k | 0.884 | 0.758 | 0.730 | 0.687 | 0.159 |
| B2 visual + h + state8 + tau/k | 0.758 | 0.692 | 0.657 | 0.696 | 0.200 |

kNN teacher retrieval on B2 features: relative 0.861, aggregate 0.847,
cosine 0.620. Teacher residual variance decomposition: between-state share
0.264, within-state 0.736.

So explicit global visual mean does not improve cross-state prediction; h-only
is better but still weak; nearest observable features are worse than ridge.
The teacher residual is mostly trajectory/state-specific rather than a simple
function of current mean visual features. Do not add a large MLP yet.

### R0-2B3 and corrected variance

Variance decomposition corrected: between-state share 0.359, within-state
0.641 (prior 0.264/0.736 used total variance as within, which was wrong).

Token-level R0 ablation (B0 h; B3a per-view pooled; B3b fixed low-rank
action-conditioned retrieval over 2x48 Resampler tokens):

| model | CV rel | val rel | val agg | val cos | Clean residual norm |
|---|---:|---:|---:|---:|---:|
| B0 | 0.632 | 0.669 | 0.653 | 0.761 | 0.148 |
| B3a | 0.657 | 0.669 | 0.652 | 0.738 | 0.166 |
| B3b | 0.696 | 0.690 | 0.737 | 0.769 | 0.113 |

Per-view pooling does not improve over B0. Fixed random low-rank token
retrieval gives slightly better cosine and lower Clean residual, but worse
relative and aggregate error. This does not support adding a learned MLP or
temporal memory; a learned retrieval module is still untested.

### R0-2B4: learned retrieval attempt (invalid optimization)

Implemented fold-internal B4a (learnable query, raw keys/values) and B4b
(learnable low-rank Q/K/V), with separate agentview/eye attention.

The first run diverged: B4a/B4b validation relative errors were enormous and
Clean predicted residual norms 0.2-0.4; the joint learned retrieval + linear
head was not stably optimized on 10 source states. These numbers are invalid
and must not be interpreted as evidence for or against retrieval. The next
fix is a closed-form ridge head on frozen retrieval features and fold-local
regularization, not a larger network.

### R0-2B4 stable fallback: normalized frozen retrieval + ridge

L2-normalized query/key attention with closed-form ridge head (no joint
retrieval/head optimization):

| model | CV rel | val rel | val agg | val cos | Clean residual |
|---|---:|---:|---:|---:|---:|
| B0 h | 0.632 | 0.669 | 0.653 | 0.761 | 0.148 |
| B3a per-view | 0.657 | 0.669 | 0.652 | 0.738 | 0.166 |
| B3b fixed random | 0.696 | 0.690 | 0.737 | 0.769 | 0.113 |
| B4fa normalized query, raw K/V | 0.657 | 0.669 | 0.652 | 0.738 | 0.166 |
| B4fb normalized low-rank QKV | 0.674 | 0.689 | 0.691 | 0.760 | 0.119 |

The L2-normalized retrieval variants do not improve cross-state residual
prediction over B0. The jointly learned B4a/B4b still diverged even with L2
normalization/gradient clipping, so those runs are invalid optimization; they
do not establish that learned retrieval is impossible. Single-frame frozen
token retrieval does not show useful conditional information under this
protocol.

### Gate 0: chunk overlap audit

Actual CoRP checkpoint and official evaluator:

```text
H = 12
E = 12 (for action in chunk:)
L = H - E = 0
num_sample_steps = 10
Euler: x_{n+1} = x_n + v_theta(x_n, tau_n) / N
```

So the current protocol has **no action-chunk leftover**. SALT-style
leftover supervision cannot be applied without changing E and re-running all
controls under the same E. The released inference repo does not define the
training target velocity sign; only the sampler sign is verified positive.

### B4 archive correction

B4 relative error had mixed Clean zero labels into the denominator, producing
huge pseudo-errors. The evaluation is now restricted to Shift samples; Clean
is reported separately as predicted residual norm. The old jointly trained B4
numbers remain invalid and are archived, not used as evidence. B4fa/B4fb are
random-projection cosine-pooling controls, not learned retrieval.

### B0 bounded Euler integration

B0 h-only ridge (r0_ridge2 coefficients) inside the full 10-step bounded
Euler sampler, fresh RGB, 20 source pairs, rho=0.10:

| branch | XYZ command | SO(3) |
|---|---:|---:|
| uncorrected Shift | 8.642 mm | 0.011883 rad |
| B0 bounded correction | 7.156 mm | 0.011302 rad |

B0 only marginally improves the integrated action; it is far below the
privileged Oracle (0.151 mm) and does not establish Clean safety. The Gate 1
first-chunk recoverability experiment was not run in this turn; it remains the
top priority before any temporal adaptation.

### Gate 1: first-chunk recoverability (E=12)

All 30 P0 Shift episodes; first 12 actions executed, then from the same
physical snapshot branch into Frozen-Shift or Frozen-Clean continuation.
P0 z0 schedule hashes were asserted.

| group | n | Shift recovery | Clean continuation |
|---|---:|---:|---:|
| Clean success / Shift fail (core) | 13 | 3 | **7** |
| both success | 8 | 6 | 6 |
| both fail | 6 | 1 | 4 |
| Clean fail / Shift success | 3 | 3 | 2 |
| all | 30 | 13 | 19 |

The first chunk leaves the core-13 episodes recoverable under Clean vision in
7/13 cases but only 3/13 under Shift. Thus roughly half of the core failures
are already irrecoverable after the first chunk even with Clean vision, while
the rest retain later-interaction headroom. This is a mixed result, not a
green light for either pure cold-start correction or pure later TTA.

### Gate 1 state clustering and Experiment B

Core-13 clustering:
- Clean-only recovery: states 34(2), 35(1), 37(1), 43(1), 45(2) = 7.
- Shift-only recovery: 35(9999), 43(7777), 45(9999) = 3.
- both fail: 43(8888), 44(9999), 48(8888) = 3.
No core episode was recovered by both. Boundary caveats: the first chunk was forced to
12 actions with no early exit, and Clean switching used reset_episode_flags(env,0);
no model_compatible/compare audit was logged. These are protocol boundaries, not proof
of physical irrecoverability.

B0 clean drift under full bounded Euler:
- Shift XYZ 8.642 mm -> 7.156 mm;
- Clean XYZ drift = 1.317–5.227 mm across states, not safe.

Experiment B (3 target states x Clean/Shift x 3 chunks, 18 boundary rows):
- mean ||dz||: Clean 13.94, Shift 12.05;
- mean ||dstate8||: Clean 0.150, Shift 0.140;
- corr(||dz||, ||dstate||) = -0.027;
- corr(||dz||, ||action_sum||) = -0.222;
- clean-fit action->state8 prediction error: Clean 0.064, Shift 1.224 (relative).
The simple chunk-boundary visual delta is not a clean domain discriminator; the
action-to-state consistency residual is more promising but the sample is tiny.

### B2: paired action-replay control

3 target states (34,43,45), same initial state, same 12 clipped actions in
Clean and Shift. Physics trajectory max difference = 0.0 for all three.
Visual Resampler tokens still differ substantially:

| state | agentview cosine | agentview RMS | eye cosine | eye RMS |
|---|---:|---:|---:|---:|
| 34 | 0.824 | 0.621 | 0.989 | 0.149 |
| 43 | 0.947 | 0.342 | 0.987 | 0.165 |
| 45 | 0.939 | 0.370 | 0.983 | 0.190 |

This cleanly localizes the Lighting 2292 shift to visual representation:
physics is identical under identical commands, while agentview tokens shift
much more than eye-in-hand. This is diagnostic only; no Clean action is used
at deployment.

### B0 split metrics

| split | n | Shift XYZ | B0 XYZ | Clean drift |
|---|---:|---:|---:|---:|
| source_fit | 10 | 8.272 mm | 6.468 mm | 1.515 mm |
| source_validation | 10 | 9.013 mm | 7.843 mm | 2.974 mm |

B0 improves validation XYZ by ~13% but Clean drift ~3 mm is unsafe.

### B2-0 parity audit

Rerun with per-step full snapshot comparison (qpos/qvel/body_pos/body_quat/eef_pos):

| state | policy_state8 maxdiff | full physics maxdiff | model_compatible |
|---|---:|---:|---:|
| 34 | 0.0 | 0.0 | false |
| 43 | 0.0 | 0.0 | false |
| 45 | 0.0 | 0.0 | false |

The full physics trajectory is identical under identical clipped actions, so
the earlier "physics identical" claim is now supported. `model_compatible`
returns false because the Clean and Lighting environments intentionally use
different scene BDDL/model files; therefore it should not be used as a
correctness gate for this pair, but it is logged.

B3-1/B3-2 were not run in this turn; only B3-0 instrumentation was completed.

### B2-0 final parity (field mapping fixed)

Snapshot keys corrected to qpos/qvel/act/body_xpos/body_xquat/eef_pos;
model compatibility uses the real "pass" key and separates state mapping from
scene model:

| state | policy_state8 maxdiff | full physics maxdiff | state mapping | scene model |
|---|---:|---:|---:|---:|
| 34 | 0.0 | 0.0 | true | true |
| 43 | 0.0 | 0.0 | true | true |
| 45 | 0.0 | 0.0 | true | true |

So under identical clipped actions the measured physics trajectory is
identical; visual agentview tokens still shift (cos .824-.947). This is the
final B2-0 evidence. B3-1/B3-2 have not been run yet.

### B3-1 pilot (instrumentation only)

3 states x Clean/Shift x 3 chunks = 18 boundary rows, H=E=12, clipped
actions and actual executed steps saved, per-view tokens saved separately.

P0/P1 ridge (train Clean, test Shift):
- P0 clean 0.143, shift 0.337
- P1 clean 0.0084, shift 0.609

P1 overfits the 9 Clean rows and is worse on Shift. With 3 states and 9
samples per domain this is not evidence for or against visual history; it is
only a protocol/instrumentation pilot. B3-2 (Clean-reference velocity
residual) was not run.

### B3 data admission: all 10 target-dev states

Controlled replay: same initial state + same 12 clipped actions in Clean and
Shift. All 10 target-dev states pass the measured physics parity:

- policy_state8 maxdiff = 0.0;
- measured physics (qpos/qvel/body_xpos/body_xquat/eef_pos) maxdiff = 0.0;
- state mapping pass;
- body/site structure pass.

Visual shift is present in all 10 but view-dependent: agentview cosine
0.824-0.947, eye-in-hand 0.961-0.989. Thus all 10 states are admissible for
B3-2 Oracle pairing. B3-1/B3-2 full runs are not yet executed.

### B3-2 pilot: same-x Clean/Shift flow residual (3 states)

3 target states x 4 z0 trajectories x 10 Euler steps x 12 action positions.
At each real Shift Euler state x, Clean and Shift velocities were evaluated on
the same x; r* = v_C - v_S. 1440 token samples.

Leave-one-state-out ridge:
- M0 (h,state8,tau,k): rel 1.712, agg 1.358, cos 0.688, mag 1.785
- M1 (+visual mean): rel 1.727, agg 1.367, cos 0.685, mag 1.797
- zero predictor: rel 1.000, cos 0.0

M0/M1 have some direction signal (cos ~0.69) but overpredict magnitude
(~1.8x) and are worse than zero on relative/aggregate error. Adding global
visual mean gives no gain. This is a 3-state pilot, not a conclusion about
action-conditioned or history-conditioned signals; no deployable correction.

### B3-2 corrected visual-feature rerun

Variable overwrite fixed: z_clean/z_shift visual tokens named separately from
v_clean/v_shift. Re-collected 3 states x 4 z0 x 10 steps x 12 positions.

Leave-one-state-out:
- M0 (h,state8,tau,k): rel 1.712, agg 1.358, cos 0.688, mag 1.785, G -0.712
- M1a (+agentview): rel 1.780, agg 1.386, cos 0.543, G -0.780
- M1w (+wrist): rel 1.380, agg 1.055, cos 0.635, mag 1.255, G -0.380
- M1both: rel 1.384, agg 1.064, cos 0.589, G -0.384
- zero: rel 1.000, G 0.0

True wrist visual features improve over M0 but still do not beat zero (G<0);
agentview features hurt. This is a 3-state pilot, not deployable.
Sampler parity: independent differentiable Euler vs sample_z max abs 0.0
normalized action (same formula); official policy.predict is not used here.

B3-1 analysis still uses planned actions; documented as not fixed.

### B3 view-token ownership test

Changing only agentview changes Resampler tokens [0:48] (mean L2 43.35) and
leaves [48:96] exactly unchanged. Changing only eye-in-hand changes [48:96]
(26.59) and leaves [0:48] unchanged. So per-view split is valid for this
checkpoint.

B3-1 analysis now uses actions_executed + executed mask, not planned actions.
Pilot remains overfit (3 states, 9 Clean rows): P0 shift .347, P1 shift .608.

### B3-2 10-state report: statistical correction (2026-11-21)

The former constant and scalar-calibration results are **retracted**. The
6fba252 analysis used `Y[test].mean(0)` for its purported train-fold constant
and test labels to estimate beta. The archived `docs/data/b3_oracle10.json`
is retained as an invalid legacy report for those two comparisons.
Do not cite its constant G_row=0.233/G_agg=0.309 or calibrated gains.

Reanalysis of the original 4800 rows uses outer-train labels only for the
constant, all preprocessing and scalar beta. Beta uses outer-train fitted
predictions (in-sample calibration, not nested validation). Policy weights
were not changed and data were not recollected. See
`docs/data/b3_oracle10_leakage_fixed.json` and fold-audit entries.

| model | G_row(all) | G_agg(all) | G_row(non-pilot 7) |
|---|---:|---:|---:|
| zero | 0.000 | 0.000 | 0.000 |
| true train-fold constant | -0.090 | 0.058 | -0.208 |
| M0 | -0.047 | 0.127 | -0.162 |
| M1a | 0.016 | 0.158 | -0.065 |
| M1w | -0.330 | -0.002 | -0.474 |
| M1both | 0.001 | 0.150 | -0.090 |

The original learned-model metrics reproduce within numerical roundoff.
The claim that a constant beats every model is FALSE after correction.
The train-only beta clips to 1 for every fold, so calibrated results equal
the raw results and no longer support the earlier claimed calibration gain.

The direct residual branch remains frozen: all raw learned models have
negative mean G_row on the seven states not used for the 3-state pilot;
there is no validated Clean No-Op or closed-loop benefit. Some aggregate
metrics are positive, so this is insufficient reliable evidence for deployment,
not proof that every residual predictor or all TTA is impossible.

These 10 states are target-dev (34,35,36,37,39,43,44,45,48,49), so the experiment
is a **privileged target-dev oracle diagnostic**, not source-only. The 34/43/45
pilot subset and the other seven states remain separately reported. Rows,
flow steps and action positions are correlated; the unit is initial state.

### Sensor zero-training signal diagnostic (40 cases, t=0)

Sensor 30 + Clean 10, frozen dev manifest. Signals at t=0:
S1 multi-z0 XYZ std; S1 gripper flip; S2 mild agentview translation sensitivity;
S4 agentview denoise action difference.

| signal | AUROC fail vs success | recall at clean FPR<=10% |
|---|---:|---:|
| S1 xyz std | 0.398 | 7.7% |
| S1 gripper flip | 0.500 | 0.0% |
| S2 perturb std | 0.400 | 0.0% |
| S4 denoise diff | 0.468 | 0.0% |

All AUCs are at or below chance in the risk direction; no signal separates
Sensor failures from Sensor successes while keeping Clean false positives low.
This rejects t=0 action-dispersion / local sensitivity / denoise-response as
standalone Sensor failure-risk signals. Do not train an Adapter from this.

### Visual quality gate at t=0 (Sensor vs Clean)

Correctly applies the official agentview sensor kernel (`sensor_transform`,
`runtime_params.noise`) before measuring image quality. 30 Sensor + 10 Clean
dev cases:

| signal | Clean vs Sensor AUC | detection at Clean max | Clean FPR |
|---|---:|---:|---:|
| agentview Laplacian variance | 1.000 | 100% | 0% |
| agentview HF ratio | 1.000 | 100% | 0% |

Sensor subtypes are blur/fog: glass 13, gaussian 7, motion 6, zoom 3, fog 1.

So a Clean No-Op / OOD gate from raw visual quality is easy and near-perfect
on Sensor. But the earlier t=0 risk signals (multi-z0 dispersion, local
perturbation, denoise response) failed to separate Sensor success vs failure
(AUC .398-.500). This confirms the split:
- Domain detection (Clean vs Sensor): easy, usable for No-Op.
- Failure prediction inside Sensor: not available from t=0 scalar signals.

### CORRECTION: Sensor t=0 signals previously missed sensor_transform

The earlier `3dccdf9` run used raw observations and did not apply the official
agentview sensor kernel. Those AUC .40-.50 numbers are invalid for Sensor.

Corrected run (official `sensor_transform`, runtime noise code, sensor seed):
30 Sensor + 10 Clean, t=0.

| signal | AUC fail vs success | recall at Clean FPR<=10% | Clean FPR |
|---|---:|---:|---:|
| S1 multi-z0 XYZ std | 0.715 | 46.2% | 10% |
| S1 multi-z0 rotation std | 0.756 | 23.1% | 10% |
| S1 gripper flip | 0.654 | 100% | 100% (unusable) |
| S2 mild translation sensitivity | 0.751 | 15.4% | 10% |
| S4 denoise action diff | 0.389 | 38.5% | 10% |

So t=0 action dispersion/sensitivity does contain a weak above-chance signal
for Sensor failure, contrary to the buggy run. It is not deployable yet:
recall at strict Clean FPR is low, and gripper flip is unusable.

Visual quality gate (Laplacian/HF) remains Clean-vs-Sensor AUC 1.0, Clean FPR 0.
