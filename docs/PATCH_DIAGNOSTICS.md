# CoRP DINO Patch Diagnostics (Phase 0, Phase 1, B0-B3)

## Claim boundary

This work is an **offline source-paired diagnostic**, not a deployable TTA
method. Only source_fit and source_validation roles are accepted. Target-dev
must not be used to select the encoder, feature normalization, thresholds, or
intervention. Target-test is not accepted by these entry points.

The Clean patches used by B2/B3 are privileged counterfactual inputs. They are
offline mediator interventions, not deployment signals.

## Why a new collector is required

The existing early/late collectors save post-Resampler visual tokens. Those
files cannot establish whether DINO patch tokens changed; substituting them
would intervene at the wrong representation layer. The new collector records
the raw per-view visual_encoder outputs before CrossAttentionResampler, with
the paired state inputs and explicit shared z0.

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

**Gate:** all identity-path errors must be below 1e-6. The report describes
runtime shape facts; it does not assume a fixed DINO architecture.

## Pair capture: raw patch features at early and late windows

~~~bash
MUJOCO_GL=osmesa python scripts/collect_patch_pairs.py \
  --matrix /path/to/libero_matrix.json \
  --checkpoint repos/CoRP/checkpoints/corp_libero.pt \
  --split-json docs/data/tta_split_libero_spatial.json \
  --suite libero_spatial --lighting-id 2292 \
  --roles source_fit,source_validation \
  --early-chunks 0,1 --late-chunks 6,7,8,9 \
  --seed 7777 --executed-k 12 \
  --out runs/patch_diag/source_patch_pairs.npz
~~~

The collector advances only the Shift source episode using Frozen CoRP actions.
At selected replan boundaries it snapshots the Shift simulator state, restores
that state in a Clean twin, and checks model, physical state, controller,
camera, and policy-state equivalence. Failed pairs remain in the artifact for
audit but are excluded by downstream analysis.

Outputs:

- source_patch_pairs.npz: per-view Clean/Shift patch tensors, state inputs,
  state/language tokens, pooled language, explicit z0, and paired actions;
- source_patch_pairs.manifest.json: split role, state/chunk/step, pair checks,
  digests, shapes, and file hashes.

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
- Clean-Clean between-state variability at the same chunk index (D_content);
- R_sep = D_style / D_content, using a source_fit-only feature scale;
- source_fit Clean prototype-distance AUROC on source_validation states;
- logistic-regression AUROC trained on source_fit paired states and tested on
  disjoint source_validation states;
- state-cluster bootstrap confidence intervals for held-out AUROC;
- average N-by-N patch self-similarity matrices, where N is the observed token
  count for that view.

The classifier is trained and tested at initial-state granularity: all
observations from a state stay in one split role. Early and late windows are
analyzed separately to avoid mixing manipulation phases.

**Interpretation:** high AUROC establishes generalizable domain information,
not causal responsibility or action benefit. Low AUROC does not prove that the
Resampler or DiT is the bottleneck. R_sep is descriptive, not a universal
pass/fail law.

## Paired Patch Replacement: B0-B3

~~~bash
python scripts/paired_patch_replacement.py \
  --pairs runs/patch_diag/source_patch_pairs.npz \
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
as the reference. B2 recovery must be compared against both B1 and the
random-content negative control B3.

**Causal interpretation guard:** B2 is a privileged offline mediator
intervention. Strong B2 recovery and weaker B3 recovery support the claim
that same-state visual patch content mediates downstream action change. They
do not establish a deployable adaptation signal and do not replace closed-loop
evaluation.

Aggregates are state-balanced: chunks are averaged within each initial state,
then states are averaged equally. Do not treat multiple chunks from one state
as independent samples.

## Execution order and stop rules

1. Phase 0 must pass before pair collection.
2. Collect source-only paired patches; inspect the manifest and pairing failures.
3. Run Phase 1. Report every view/window independently.
4. Run B0-B3 on the same pairing-pass rows and explicit z0 values.
5. Consider Phase 2 visual alignment only if B2 restores Resampler/action
   outputs materially better than B1 and B3, while Clean identity remains
   acceptable.

No training, adapter update, OT, AdaIN, target-dev rollout, or target-test
access is part of these steps.
